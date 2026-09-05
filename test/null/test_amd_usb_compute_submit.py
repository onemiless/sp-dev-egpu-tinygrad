import unittest
from types import SimpleNamespace

from tinygrad.runtime.ops_amd import AMDComputeQueue


class FakeRing:
  def __init__(self, size, events): self.data, self.events = [None] * size, events
  def __len__(self): return len(self.data)
  def __setitem__(self, index, value):
    vals = list(value) if isinstance(index, slice) else value
    self.events.append(("ring", index, vals))
    self.data[index] = vals


class FakeQueue:
  def __init__(self, ring_size, put_value, events): self.ring, self.put_value, self.events = FakeRing(ring_size, events), put_value, events
  def signal_doorbell(self, _dev): self.events.append(("doorbell", self.put_value))


class FakeDevice:
  def __init__(self, usb, ring_size, put_value, events): self.compute_queue, self.usb = FakeQueue(ring_size, put_value, events), usb
  def is_usb(self): return self.usb


def submit(cmds, *, usb=True, ring_size=8, put_value=0):
  events = []
  dev = FakeDevice(usb, ring_size, put_value, events)
  queue = SimpleNamespace(indirect_cmd=cmds, binded_device=dev, _q=cmds, dev=SimpleNamespace(xccs=1))
  AMDComputeQueue._submit(queue, dev)
  return dev.compute_queue, events


class TestAMDUSBComputeSubmit(unittest.TestCase):
  def test_contiguous_submit_is_one_transfer(self):
    queue, events = submit([1, 2, 3, 4], put_value=2)
    self.assertEqual(queue.ring.data, [None, None, 1, 2, 3, 4, None, None])
    self.assertEqual(queue.put_value, 6)
    self.assertEqual(events, [("ring", slice(2, 6), [1, 2, 3, 4]), ("doorbell", 6)])

  def test_submit_ending_at_ring_boundary_is_one_transfer(self):
    queue, events = submit([1, 2, 3, 4], put_value=4)
    self.assertEqual(queue.ring.data, [None, None, None, None, 1, 2, 3, 4])
    self.assertEqual(events, [("ring", slice(4, 8), [1, 2, 3, 4]), ("doorbell", 8)])

  def test_wrapped_submit_is_two_ordered_transfers(self):
    queue, events = submit([1, 2, 3, 4], put_value=6)
    self.assertEqual(queue.ring.data, [3, 4, None, None, None, None, 1, 2])
    self.assertEqual(queue.put_value, 10)
    self.assertEqual(events, [
      ("ring", slice(6, 8), [1, 2]),
      ("ring", slice(0, 2), [3, 4]),
      ("doorbell", 10),
    ])

  def test_absolute_put_value_is_reduced_only_for_ring_index(self):
    queue, events = submit([1, 2, 3], put_value=10)
    self.assertEqual(queue.ring.data, [None, None, 1, 2, 3, None, None, None])
    self.assertEqual(queue.put_value, 13)
    self.assertEqual(events[-1], ("doorbell", 13))

  def test_native_pcie_keeps_individual_dword_writes(self):
    queue, events = submit([1, 2, 3, 4], usb=False, put_value=6)
    self.assertEqual(queue.ring.data, [3, 4, None, None, None, None, 1, 2])
    self.assertEqual(events, [
      ("ring", 6, 1), ("ring", 7, 2), ("ring", 0, 3), ("ring", 1, 4), ("doorbell", 10),
    ])


if __name__ == "__main__": unittest.main()
