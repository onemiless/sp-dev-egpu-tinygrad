import struct, unittest
from types import SimpleNamespace

from tinygrad.runtime.ops_amd import AMDAllocator


CHUNK = 0x40000 - 4
FENCE = 0xA800


class FakeBuffer:
  def __init__(self, va_addr:int, size:int): self.va_addr, self.size = va_addr, size
  def offset(self, offset:int=0, size:int|None=None): return FakeBuffer(self.va_addr + offset, size or self.size - offset)


class FakeCopyQueue:
  def __init__(self, harness): self.harness = harness
  def wait(self, _signal, value):
    self.harness.events.append(("queue_wait", value))
    return self
  def q(self, *values): self.harness.events.append(("sentinel_poll", values[3]))
  def copy(self, dest, src, size):
    self.harness.copy_sizes.append(size)
    self.harness.events.append(("copy", dest.va_addr, src.va_addr, size))
    return self
  def write(self, _buf, value, b64=False):
    self.harness.events.append(("fence_write", value, b64))
    return self
  def signal(self, _signal, value):
    self.harness.events.append(("queue_signal", value))
    return self
  def submit(self, _dev):
    self.harness.queue_submits += 1
    self.harness.events.append(("queue_submit",))
    return self


class FakeController:
  def __init__(self, harness):
    self.harness, self.fence = harness, 0
    self.landed, self.blocked = set(), set(harness.delayed_sequences)
    self.usb = FakeUSB(harness, self)

  def _advance_drain(self):
    while self.fence in self.landed and self.fence not in self.blocked:
      seq = self.fence
      self.landed.remove(seq)
      self.fence += 1
      self.harness.events.append(("drained", seq))

  def land(self, seq:int):
    self.landed.add(seq)
    self._advance_drain()

  def read(self, address:int, length:int):
    assert (address, length) == (FENCE, 8)
    value = self.fence
    self.harness.events.append(("sync_e4", value))
    # A delayed drain is released only after returning one stale sample. This
    # makes AMDAllocator exercise its synchronous wait_drain fallback.
    if self.fence in self.blocked and self.fence in self.landed:
      self.blocked.remove(self.fence)
      self._advance_drain()
    return value.to_bytes(8, "little")


class FakeUSB:
  def __init__(self, harness, controller):
    self.harness, self.controller = harness, controller
    self.next_tag, self.pending, self.armed_slot = 1, {}, None

  def _submit(self, kind, metadata=None):
    tag, self.next_tag = self.next_tag, self.next_tag + 1
    self.pending[tag] = (kind, metadata)
    return tag

  def control_write_async(self, request:int, value:int=0, index:int=0, data:bytes=b"", timeout:int=1000):
    assert request == 0xF2 and not value & 0x8000 and not data
    tag = self._submit("f2", (index & 0xFF, value * 512))
    self.harness.events.append(("f2_submit", index & 0xFF, value * 512, tag))
    return tag

  def control_read_async(self, request:int, length:int, value:int=0, index:int=0, timeout:int=1000):
    assert (request, length, value, index) == (0xE4, 8, FENCE, 0)
    result = bytearray(8)
    tag = self._submit("e4", result)
    self.harness.events.append(("e4_submit", tag))
    return tag, memoryview(result)

  def bulk_write_async(self, payload:memoryview, timeout:int=10000):
    assert self.armed_slot is not None
    wire = bytes(payload)
    assert len(wire) == self.armed_slot[1]
    sentinel = struct.unpack_from("<I", wire, len(wire) - 4)[0]
    assert sentinel >> 24 == 0x51
    seq = sentinel & 0xFFFFFF
    tag = self._submit("ep02", seq)
    self.harness.ep02_sequences.append(seq)
    self.harness.events.append(("ep02_submit", seq, self.armed_slot[0], len(wire), tag))
    self.armed_slot = None
    return tag

  def bulk_wait(self, tag):
    if tag is None: return
    kind, metadata = self.pending.pop(tag)
    self.harness.events.append((f"{kind}_wait", tag))
    if self.harness.error_kind == kind: raise RuntimeError(f"{kind} async failed")
    if kind == "f2": self.armed_slot = metadata
    elif kind == "e4": metadata[:] = self.controller.fence.to_bytes(8, "little")
    elif kind == "ep02": self.controller.land(metadata)


class CopyinHarness:
  def __init__(self, *, delayed_sequences=(), error_kind=None):
    self.delayed_sequences, self.error_kind = set(delayed_sequences), error_kind
    self.events, self.copy_sizes, self.ep02_sequences = [], [], []
    self.queue_submits = 0
    self.controller = FakeController(self)
    sdma = SimpleNamespace(
      SDMA_OP_POLL_REGMEM=0x10,
      SDMA_PKT_POLL_REGMEM_HEADER_FUNC=lambda value: value << 4,
      SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL=lambda value: value << 8,
      SDMA_PKT_POLL_REGMEM_DW5_INTERVAL=lambda value: value,
      SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT=lambda value: value << 16,
    )
    iface = SimpleNamespace(pci_dev=SimpleNamespace(usb=self.controller), sys_buf=FakeBuffer(0x820000, 0x1000))
    self.dev = SimpleNamespace(iface=iface, timeline_signal=object(), timeline_value=1, sdma=sdma)
    self.dev.is_usb = lambda: True
    self.dev.hw_copy_queue_t = lambda: FakeCopyQueue(self)
    def next_timeline():
      value = self.dev.timeline_value
      self.dev.timeline_value += 1
      return value
    self.dev.next_timeline = next_timeline

    self.allocator = AMDAllocator.__new__(AMDAllocator)
    self.allocator.dev = self.dev
    self.allocator.b = [FakeBuffer(0x200000, 0x80000)]
    self.allocator._usb_seq = 0
    # Keep one backing object per view, as the real alloc_cbuffer pair does.
    self.allocator._usb_stage = [(backing, memoryview(backing)) for backing in (bytearray(0x40000), bytearray(0x40000))]
    self.allocator._usb_wins = (FakeBuffer(0x200000, 0x40000), FakeBuffer(0x240000, 0x40000))

  def copyin(self, size:int):
    AMDAllocator._copyin(self.allocator, FakeBuffer(0x10000000, size), memoryview(bytearray(size)))

  def count(self, event): return sum(item[0] == event for item in self.events)


class TestAMDUSBCopyinProtocol(unittest.TestCase):
  def test_e4_is_submitted_only_when_a_bounce_window_is_reused(self):
    observed = []
    for size in (1, CHUNK, CHUNK + 1, 2 * CHUNK, 2 * CHUNK + 1):
      harness = CopyinHarness()
      harness.copyin(size)
      observed.append((harness.count("f2_submit"), harness.count("ep02_submit"), harness.count("e4_submit")))
      self.assertEqual(harness.queue_submits, 1)
      self.assertEqual(sum(harness.copy_sizes), size)
    self.assertEqual(observed, [(1, 1, 0), (1, 1, 0), (2, 2, 0), (2, 2, 0), (3, 3, 1)])

  def test_separate_two_and_one_chunk_copyins_do_not_probe_for_reuse(self):
    harness = CopyinHarness()
    harness.copyin(2 * CHUNK)
    harness.copyin(1)
    self.assertEqual(harness.count("f2_submit"), 3)
    self.assertEqual(harness.count("ep02_submit"), 3)
    self.assertEqual(harness.count("e4_submit"), 0)
    self.assertEqual(harness.queue_submits, 2)

  def test_delayed_drain_blocks_the_reused_window(self):
    harness = CopyinHarness(delayed_sequences={0})
    harness.copyin(2 * CHUNK + 1)
    stale_read = harness.events.index(("sync_e4", 0))
    drain = harness.events.index(("drained", 0))
    reuse = next(i for i, event in enumerate(harness.events) if event[:2] == ("ep02_submit", 2))
    self.assertLess(stale_read, drain)
    self.assertLess(drain, reuse)
    self.assertEqual(harness.controller.fence, 3)

  def test_f2_async_error_prevents_payload_submit(self):
    harness = CopyinHarness(error_kind="f2")
    with self.assertRaisesRegex(RuntimeError, "f2 async failed"): harness.copyin(1)
    self.assertEqual(harness.count("ep02_submit"), 0)

  def test_reuse_e4_async_error_prevents_reused_payload_submit(self):
    harness = CopyinHarness(error_kind="e4")
    with self.assertRaisesRegex(RuntimeError, "e4 async failed"): harness.copyin(2 * CHUNK + 1)
    self.assertEqual(harness.ep02_sequences, [0, 1])

  def test_ep02_async_error_is_propagated(self):
    harness = CopyinHarness(error_kind="ep02")
    with self.assertRaisesRegex(RuntimeError, "ep02 async failed"): harness.copyin(1)
    self.assertEqual(harness.ep02_sequences, [0])


if __name__ == "__main__": unittest.main()
