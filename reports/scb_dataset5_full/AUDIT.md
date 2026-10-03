# SCB-Dataset5 Full — Physical Annotation Audit

**Audit Date**: 2026-10-02  

## 1. Actual Physically Present Classes

| Class Name | Physical Annotations | Unique Images | Share (%) | Verified Sources | Contact Sheet |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `read` | 24,078 | 3,878 | 21.5028% | SCB5-Handrise-Read-write-2024-9-17 (id 1) | [samples](class_samples/read.jpg) |
| `hand-raising` | 13,453 | 4,064 | 12.0142% | SCB5-Handrise-Read-write-2024-9-17 (id 0) | [samples](class_samples/hand-raising.jpg) |
| `stand` | 12,969 | 7,602 | 11.5819% | SCB5-Stand-2024-9-17 (id 0) | [samples](class_samples/stand.jpg) |
| `teacher` | 11,738 | 11,163 | 10.4826% | SCB5-BlackBoard-Sreen-Teacher (id 2), SCB5-Teacher-2024-9-17 (id 0) | [samples](class_samples/teacher.jpg) |
| `talk` | 11,012 | 9,078 | 9.8343% | SCB5-Talk-2024-9-17 (id 0), SCB5-Talk-Teacher-Behavior-2024-9-17 (id 0) | [samples](class_samples/talk.jpg) |
| `write` | 9,841 | 1,721 | 8.7885% | SCB5-Handrise-Read-write-2024-9-17 (id 2) | [samples](class_samples/write.jpg) |
| `screen` | 6,961 | 4,709 | 6.2165% | SCB5-BlackBoard-Screen (id 1), SCB5-BlackBoard-Sreen-Teacher (id 1) | [samples](class_samples/screen.jpg) |
| `answer` | 6,484 | 6,350 | 5.7905% | SCB5-Talk-Teacher-Behavior-2024-9-17 (id 2), SCB5-Teacher-Behavior-2024-9-17 (id 1) | [samples](class_samples/answer.jpg) |
| `discuss` | 5,392 | 864 | 4.8153% | SCB5-Discuss-2024-9-17 (id 0) | [samples](class_samples/discuss.jpg) |
| `blackBoard` | 4,160 | 3,988 | 3.7151% | SCB5-BlackBoard-Screen (id 0), SCB5-BlackBoard-Sreen-Teacher (id 0) | [samples](class_samples/blackBoard.jpg) |
| `guide` | 2,864 | 2,690 | 2.5577% | SCB5-Talk-Teacher-Behavior-2024-9-17 (id 1), SCB5-Teacher-Behavior-2024-9-17 (id 0) | [samples](class_samples/guide.jpg) |
| `blackboard-writing` | 1,868 | 1,796 | 1.6682% | SCB5-Talk-Teacher-Behavior-2024-9-17 (id 4), SCB5-Teacher-Behavior-2024-9-17 (id 3) | [samples](class_samples/blackboard-writing.jpg) |
| `On-stage interaction` | 1,156 | 1,138 | 1.0324% | SCB5-Talk-Teacher-Behavior-2024-9-17 (id 3), SCB5-Teacher-Behavior-2024-9-17 (id 2) | [samples](class_samples/On-stage_interaction.jpg) |

## 2. Expected vs Actual Physical Status

| Expected Concept | Physically Present in Full SCB5 | Matched Physical Class | Notes |
| :--- | :---: | :--- | :--- |
| `hand-raising` | **YES** | `hand-raising` | Physically verified |
| `read` | **YES** | `read` | Physically verified |
| `write` | **YES** | `write` | Physically verified |
| `bow head` | **NO** | `MISSING` | Not present in local annotations |
| `turn head` | **NO** | `MISSING` | Not present in local annotations |
| `BowHead` | **NO** | `MISSING` | Not present in local annotations |
| `TurnHead` | **NO** | `MISSING` | Not present in local annotations |
| `talk` | **YES** | `talk` | Physically verified |
| `stand` | **YES** | `stand` | Physically verified |
| `discuss` | **YES** | `discuss` | Physically verified |
| `using phone` | **NO** | `MISSING` | Not present in local annotations |
| `leaning on desk` | **NO** | `MISSING` | Not present in local annotations |
| `teacher` | **YES** | `teacher` | Physically verified |
| `screen` | **YES** | `screen` | Physically verified |
| `blackboard` | **YES** | `blackBoard` | Physically verified |
| `guide` | **YES** | `guide` | Physically verified |
| `answer` | **YES** | `answer` | Physically verified |
