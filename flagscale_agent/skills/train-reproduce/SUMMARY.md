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

# Reproduce — Summary

Reproduce training results from open-source implementations to establish a verified baseline before migrating to FlagScale.

**Load when**: reproducing a paper's training results, establishing a reference baseline for precision alignment, or validating that a source implementation works before porting.

Key concept: IMMUTABLE parameters (model arch, tokenizer, optimizer, loss, data) vs ADAPTABLE parameters (parallelism, hardware, batch schedule). "Reproduce" = strict immutable params. "Verify" = quick pipeline check.
