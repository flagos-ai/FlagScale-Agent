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

# Operational Discipline — Summary

General operational rules for FlagScale infrastructure work: pitfall recall, reading strategy, shell safety, remote execution posture (ssh → docker), environment awareness, root cause diagnosis, and the experiment ledger gate.

**Load when**: starting infrastructure work on a new server, executing remote/docker operations, debugging shell/environment issues, or before training launches.

Key rules: recall pitfalls from memory BEFORE the error; read complete files before implementing; never run the same command twice; same-object read/write calls never batch in parallel; docker exec needs `-i` for stdin and three checkpoints (landed / version / effect) before trusting a remote change; record every training attempt in the memory ledger before launch; back up before irreversible git operations. For training-specific operations, use train-run skill instead.
