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

Launch, stop, and manage FlagScale distributed training jobs on GPU servers.

**Load when**: launching training, stopping a run, checking GPU availability, or debugging launch failures.

Covers: server connection, environment checks, GPU availability, preflight validation (dependencies + data + config arithmetic), training launch via FlagScale CLI, stop commands, post-launch monitoring (stderr first!), log directory structure, and quick verification paths.
