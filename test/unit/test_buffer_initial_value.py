import pickle
from unittest.mock import patch

from tinygrad.device import Buffer
from tinygrad.dtype import dtypes


class FakeDevice:
  def __init__(self, usb: bool): self.usb = usb
  def is_usb(self) -> bool: return self.usb


class FakeAllocator:
  def __init__(self, usb: bool): self.dev = FakeDevice(usb)
  def free(self, *_args): pass


def construct_with_fake_allocator(payload: bytearray, *, usb: bool):
  captured = {}
  allocator = FakeAllocator(usb)

  def allocate(self, opaque=None):
    self.allocator = allocator
    self._bufs[self.device] = opaque if opaque is not None else object()
    return self

  def copy_from(self, src):
    captured["source_obj"] = src._buf.obj
    captured["source_bytes"] = bytes(src._buf)
    return self

  with patch.object(Buffer, "allocate", allocate), patch.object(Buffer, "copy_from", copy_from):
    Buffer("AMD", len(payload), dtypes.uint8, initial_value=pickle.PickleBuffer(payload))
  return captured


def test_usb_amd_picklebuffer_uses_original_host_storage():
  payload = bytearray(range(64))
  captured = construct_with_fake_allocator(payload, usb=True)
  assert captured["source_obj"] is payload
  assert captured["source_bytes"] == payload


def test_non_usb_picklebuffer_keeps_owned_copy():
  payload = bytearray(range(64))
  captured = construct_with_fake_allocator(payload, usb=False)
  assert captured["source_obj"] is not payload
  assert captured["source_bytes"] == payload
