import json
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import torch


@dataclass
class Schedule:
    skip_first: int = 0
    wait: int = 0
    warmup: int = 0
    active: int = 1
    repeat: int = 0


class RecordMode(Enum):
    Skip = 1
    Observe = 2
    Record = 3


class Profile:
    def __init__(self, model, schedule: Optional[Schedule] = None, name="model"):
        self.name_map = self._build_name_map(model, name)
        self.hooks_handle_list = []
        self.events = []
        self.schedule = schedule or Schedule()
        self.cycle_length = (
            self.schedule.wait + self.schedule.warmup + self.schedule.active
        )
        self.iter_step = 0
        self.starting_event = None
        self.recording_mode = Schedule.Skip

    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        if self.recording_mode == RecordMode.Skip:
            return
        self._register_event(
            self.name_map[module] + "_fwd",
            "B",
            torch.cuda.Event(enable_timing=True),
            self.recording_mode == RecordMode.Record,
        )

    def _forward_post_hook(self, module, inputs, outputs):
        if self.recording_mode == RecordMode.Skip:
            return
        self._register_event(
            self.name_map[module] + "_fwd",
            "E",
            torch.cuda.Event(enable_timing=True),
            self.recording_mode == RecordMode.Record,
        )

    def _backward_pre_hook(self, module, grad_output):
        if self.recording_mode == RecordMode.Skip:
            return
        self._register_event(
            self.name_map[module] + "_bwd",
            "B",
            torch.cuda.Event(enable_timing=True),
            self.recording_mode == RecordMode.Record,
        )

    def _backward_post_hook(self, module, grad_input, grad_output):
        if self.recording_mode == RecordMode.Skip:
            return
        self._register_event(
            self.name_map[module] + "_bwd",
            "E",
            torch.cuda.Event(enable_timing=True),
            self.recording_mode == RecordMode.Record,
        )

    def __enter__(self):
        self.recording_mode = self._update_recording_mode()
        return self

    def __exit__(self, type, value, traceback):
        self._remove_hooks()
        torch.cuda.synchronize()

    def step(self):
        old_mode = self.recording_mode
        self.recording_mode = self._update_recording_mode()
        if self.recording_mode != old_mode:
            print(self.recording_mode)

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        torch.cuda.synchronize()
        perfetto_events = []
        pid = os.getpid()
        tid = threading.get_native_id()

        for event in self.events:
            if event["register"]:
                perfetto_events.append(
                    {
                        "name": event["name"],
                        "ph": event["phase"],
                        "ts": int(self.starting_event.elapsed_time(event["event"]) * 1000),
                        "pid": pid,
                        "tid": tid,
                    }
                )
        with open(path, "w") as file:
            json.dump({"traceEvents": perfetto_events}, file)

    def _register_event(
        self,
        name: str,
        phase: Literal["B", "E"],
        event_value: torch.cuda.Event,
        register: bool,
    ) -> None:
        event_value.record()
        self.events.append(
            {
                "name": name,
                "phase": phase,
                "event": event_value,
                "register": register,
            }
        )

    def _set_up_hooks(self):
        print("Starting")
        for modules in self.name_map.keys():
            self.hooks_handle_list.append(
                modules.register_forward_pre_hook(self._forward_pre_hook)
            )
            self.hooks_handle_list.append(
                modules.register_forward_hook(self._forward_post_hook)
            )
            self.hooks_handle_list.append(
                modules.register_full_backward_pre_hook(self._backward_pre_hook)
            )
            self.hooks_handle_list.append(
                modules.register_full_backward_hook(self._backward_post_hook)
            )
        self.starting_event = torch.cuda.Event(enable_timing=True)
        self.starting_event.record()

    def _remove_hooks(self):
        if self.hooks_handle_list:
            for hook in self.hooks_handle_list:
                hook.remove()
            self.hooks_handle_list.clear()

    def _update_recording_mode(self) -> RecordMode:
        cycle_step = self.iter_step - self.schedule.skip_first
        self.iter_step += 1
        if cycle_step < 0:
            return RecordMode.Skip
        if self.schedule.repeat > 0 and cycle_step >= self.schedule.repeat * self.cycle_length:
            self._remove_hooks()
            return RecordMode.Skip
        if cycle_step == 0:
            self._set_up_hooks()
        cycle_pos = cycle_step % self.cycle_length
        if cycle_pos < self.schedule.wait:
            return RecordMode.Skip
        if cycle_pos < self.schedule.wait + self.schedule.warmup:
            return RecordMode.Observe
        return RecordMode.Record
