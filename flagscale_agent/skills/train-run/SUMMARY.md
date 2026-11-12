<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Train-Run — Summary

Launch, monitor, stop, and verify FlagScale training from a YAML recipe.

**Load when**: launching or stopping training, running bounded single-host Megatron trials, or diagnosing launch failures.

Uses `flagscale train -c ...` with device-specific NVIDIA and Ascend references. Includes scripts for bounded trials and Megatron log analysis, preserving execution evidence and measurement reports for the calling tuning workflow.
