# ExamGuard Vision — Security Architecture & Threat Model

## 1. Executive Summary & Security Objectives
ExamGuard Vision operates as an AI-assisted proctoring appliance on edge hardware. This document outlines the security architecture, threat model, cryptographic envelopes, key management, role-based access control (RBAC), and operational boundaries established for the production foundation.

### Security Posture & Reality
- **Current Target Environment**: Controlled Windows pilot node (ASUS TUF Gaming A17 / RTX 3050 Laptop GPU) running on localhost.
- **Production Status**: `READY_FOR_PRODUCTION = NO`, `READY_FOR_CONTROLLED_SECURE_PILOT = YES` (upon successful validation).
- **Core Security Principles**: Defense in depth, least privilege, zero default credentials, at-rest cryptographic confidentiality and integrity, and explicit non-repudiation via tamper-evident audit logging.

---

## 2. Threat Model

### 2.1 Protected Assets
1. **Student / Candidate Privacy Media**:
   - High-resolution cropped snapshots of candidate postures and behaviors.
   - 10-second rolling video clips (5s pre-roll + 5s post-roll) in MP4 format.
2. **Monitoring Sessions & State**:
   - Live camera feeds, active session metadata, room assignments.
   - Observable event state machine, risk metrics, and anomaly timestamps.
3. **Proctor Review Decisions**:
   - Human adjudications (CONFIRM / DISMISS), review notes, reviewer identities.
4. **Audit Logs & Provenance**:
   - Append-only audit log records chained with cryptographic SHA-256 hashes.
5. **Database & Backups**:
   - SQLite metadata database (`examguard.sqlite3`).
   - Transactionally consistent and encrypted backup bundles.
6. **Security Credentials & Keys**:
   - Master evidence encryption keys.
   - User password hashes (Argon2id).
   - Authenticated session tokens.
   - RTSP streaming credentials / secrets.

### 2.2 Threat Vectors & Mitigations
| Threat Vector | Potential Impact | Architecture Mitigation |
| :--- | :--- | :--- |
| **Unauthorized Local/Network UI Access** | Unauthorized monitoring or viewing of student behavior | Authentication required for all endpoints except `GET /health`. Unauthenticated requests redirect to `/login`. |
| **Default Credential Exploitation** | Instant compromise via known defaults (`admin/admin`) | **Zero hardcoded credentials**. First-run setup mode (`/setup`) forces creation of custom admin; closes permanently after initialization. |
| **Stolen Evidence Files from Disk** | Privacy violation if disk or storage folder is exfiltrated | **AES-256-GCM encryption at rest** for all snapshots, clips, and manifests. Encrypted with master key protected via Windows DPAPI. |
| **Stolen Backup Copies** | Offsite exfiltration of SQLite DB and evidence files | **Encrypted backup archives** wrapped with Argon2id/scrypt key-derivation function using a user-supplied recovery passphrase. |
| **Password Guessing / Brute-Force** | Account takeover via dictionary attacks | Rate-limiting & lockout: 5 failed attempts locks the account temporarily (15 minutes). Hashes stored using Argon2id. |
| **Session Hijacking / Token Theft** | Impersonation of logged-in proctor/admin | Cryptographically random tokens (`secrets.token_urlsafe(32)`). Server stores only SHA-256 hash. `HttpOnly`, `SameSite=Lax`, server-side revocation on logout. |
| **Cross-Site Request Forgery (CSRF)** | Unauthorized state changes via malicious web pages | Session-bound CSRF token required on all state-changing routes (`POST`, `PATCH`, `DELETE`). |
| **Role Escalation** | Invigilator performing administrative tasks or tampering | Centralized RBAC resolver (`src/security/permissions.py`). Server-side validation on every sensitive route. |
| **Path Traversal / Arbitrary File Read** | Leaking system or other student files | Strict sanitization of relative paths, boundary containment checks inside `storage/sessions/`, ID-based resolution. |
| **Secret / Key Leakage in Git or Logs** | Permanent credential compromise | Explicit `.gitignore` for keys (`storage/security/`, `*.dpapi`), automatic log redactor for passwords, tokens, and RTSP URLs. |
| **Cross-Camera Event Bleed / Track Collision** | False accusations from unrelated camera tracks | Isolation by composite key `(camera_id, track_id)`. Independent inference queues, temporal buffers, and ring buffers. |
| **Database Corruption during Migration** | Data loss of historical sessions or evidence | Transactional SQLite schema migrations (`schema_meta`), WAL mode, `PRAGMA foreign_keys = ON`. |

### 2.3 Honest Security Boundaries
- **In-Scope Protection**:
  - Offline theft of raw storage folders, unauthenticated network devices, rogue browser tabs, unauthorized proctors, replay tampering, and accidental credential leakage.
- **Out-of-Scope (Cannot Prevent)**:
  - An attacker with root / Windows Administrator / SYSTEM privileges on the physical machine who can dump process memory, inspect DPAPI decryption in-flight, or patch Python bytecode.
  - Plain HTTP transmission over untrusted networks (plain HTTP on localhost is acceptable for pilot; remote deployments strictly require TLS/HTTPS).

---

## 3. Authentication Architecture

### 3.1 First-Run Setup Mode
1. On startup, the persistence engine queries the `users` table.
2. If `count(users) == 0`:
   - System enters **First-Run Setup Mode**.
   - Launcher navigates browser to `/setup`.
   - Admin account creation form requests: `username`, `display_name`, `password`, `confirm_password`.
3. Once an admin account is created:
   - `/setup` becomes disabled and responds with HTTP 403 / redirect to `/login`.
   - Anonymous administrative user creation is permanently barred.

### 3.2 Password Hashing & Policy
- **Algorithm**: Argon2id (`argon2-cffi`).
  - Time cost: 2 iterations
  - Memory cost: 64 MB (65536 KiB)
  - Parallelism: 2 threads
  - Salt: 16 bytes cryptographically random
- **Verification**: Constant-time verification provided by Argon2 library.
- **Policy**:
  - Minimum 12 characters.
  - Permits uppercase, lowercase, numbers, symbols, and Unicode characters.
  - Enforced server-side with localized Vietnamese UI feedback.

### 3.3 Server-Side Auth Sessions
- Tokens generated using `secrets.token_urlsafe(32)` (256 bits of entropy).
- Database table `auth_sessions`:
  - `auth_session_id`: UUID PK
  - `user_id`: Foreign key to `users`
  - `token_hash`: SHA-256 hash of token (plaintext token is never stored in DB)
  - `created_at`, `expires_at`, `last_seen_at`, `revoked_at`
- **Cookie Configuration**:
  - Name: `eg_session`
  - `HttpOnly = True`
  - `SameSite = "lax"`
  - `Path = "/"`
  - `Secure`: Configurable (False on localhost HTTP, True when HTTPS enabled).
- **Session Duration**: 8 hours default lifetime; explicit logout revokes server-side session immediately.

### 3.4 CSRF Protection
- Server issues a random CSRF token tied to the active session.
- Client passes token in HTTP header `X-CSRF-Token` or form payload.
- All non-safe methods (`POST`, `PATCH`, `DELETE`) validate CSRF before processing.

---

## 4. Role-Based Access Control (RBAC)

### 4.1 Roles & Vietnamese Display
- `ADMIN` (`Quản trị viên`): Full system control, user management, security status, camera configuration, backups, audit review.
- `INVIGILATOR` (`Giám thị`): Live monitoring, real-time alert review (confirm/dismiss), current session management, camera switching.
- `REVIEWER` (`Người rà soát`): Historical session browsing, post-exam event adjudication, evidence playback.
- `VIEWER` (`Chỉ xem`): Read-only observation, cannot confirm/dismiss or alter system state.

### 4.2 Permission Matrix
| Capability | Permission Name | ADMIN | INVIGILATOR | REVIEWER | VIEWER |
| :--- | :--- | :---: | :---: | :---: | :---: |
| View Live Monitoring | `monitor:view` | Yes | Yes | No | Yes |
| Manage Live Cameras | `cameras:manage` | Yes | No | No | No |
| Switch Live Camera View | `cameras:view` | Yes | Yes | No | Yes |
| Confirm / Dismiss Events | `events:adjudicate` | Yes | Yes | Yes | No |
| View Evidence (Snapshots/Clips) | `evidence:view` | Yes | Yes | Yes | Yes |
| View History Sessions | `history:view` | Yes | Yes | Yes | Yes |
| Edit Session Metadata | `session:edit` | Yes | Yes | No | No |
| Manage System Users | `users:manage` | Yes | No | No | No |
| Create / Export Backups | `backup:create` | Yes | No | No | No |
| View System Security & Audit | `audit:view` | Yes | No | No | No |

---

## 5. Cryptography & Key Management

### 5.1 Evidence Encryption at Rest (AES-256-GCM)
- Every persisted evidence artifact (`snapshot.jpg.enc`, `clip.mp4.enc`, `manifest.json.enc`) is encrypted prior to writing to disk.
- **Binary Envelope Specification (`EGE1`)**:
  - `Magic / Version`: 4 bytes ASCII `EGE1`
  - `Key ID`: 16 bytes UTF-8 (identifier of the encryption key)
  - `Nonce`: 12 bytes cryptographically random (96-bit standard for GCM)
  - `Payload`: Ciphertext + 16-byte GCM authentication tag
- **Additional Authenticated Data (AAD)**:
  - Canonical JSON string binding: `{"session_id": "...", "event_id": "...", "evidence_id": "...", "evidence_type": "..."}`
  - Prevents ciphertext substitution or tampering across different events.

### 5.2 Key Management Architecture (`KeyProvider`)
- **Abstract Base**: `KeyProvider` interface providing `get_key(key_id)` and `get_current_key()`.
- **Windows Implementation (`WindowsDPAPIKeyProvider`)**:
  - On first boot, generates a 256-bit random master key (`secrets.token_bytes(32)`).
  - Encrypts master key using Windows Data Protection API (`CryptProtectData`) scoped to `CurrentUser`.
  - Saves encrypted blob to `storage/security/master-key.dpapi`.
  - Restricts NTFS file permissions via `icacls` to current user only (`(R,W)`).
  - Raw key is never written to disk unencrypted.
- **DPAPI Portability Limitation**:
  - Because DPAPI encrypts using Windows user-profile keys, copying `storage/security/master-key.dpapi` alone to another computer will NOT permit decryption.
  - Cross-machine portability is solved via **Portable Backup Key Wrapping** (see Section 5.4).
- **Test / Portable Implementations**:
  - `InMemoryKeyProvider` (for isolated automated testing).
  - `EnvironmentKeyProvider` (`EXAMGUARD_MASTER_KEY` environment variable).

### 5.3 Legacy Plaintext Evidence Migration
- `encryption_state` column distinguishes:
  - `LEGACY_PLAINTEXT` (unencrypted legacy recordings)
  - `ENCRYPTED_V1` (AES-256-GCM protected)
- Migration utility `tools/security/migrate_evidence_encryption.py` supports:
  - `--dry-run`: Inspects and previews eligible files.
  - `--apply`: Encrypts file to temporary destination, verifies decryption and SHA-256 provenance, atomically renames file, updates database, and safely unlinks plaintext original.

### 5.4 Encrypted Portable Backups & Memory-Only Snapshot Architecture
- **Zero Plaintext Snapshot on Disk (`NO_PLAINTEXT_BACKUP_DB_ON_DISK = YES`)**:
  - Live SQLite DB snapshot is taken in-memory using `sqlite3.Connection.backup()` to a temporary `:memory:` connection.
  - Serialized to byte buffer in RAM via `sqlite3.Connection.serialize()`.
  - Byte buffer is immediately encrypted via AES-256-GCM before writing to storage. No unencrypted `temp.db` touches the filesystem.
  - WAL journal mode offsets (18 & 19) are normalized to rollback mode (1, 1) in memory, ensuring standalone consistency during in-memory deserialization without searching for disk WAL logs.
- **Atomic Encrypted Disk Write**:
  - Ciphertext is written to a uniquely named temporary file (`<archive_name>.examguard-backup-tmp`).
  - Upon successful stream completion and SHA-256 computation, the file is atomically renamed (`os.replace`) to its final `.enc` destination.
  - Interrupted operations leave behind no valid backup entry.
- **Startup & Runtime Stale Temp Cleanup**:
  - The persistence engine executes scoped cleanup targeting orphaned `*.examguard-backup-tmp` files.
  - ExamGuard-owned temp artifacts are safely removed and audited (`BACKUP_STALE_TEMP_CLEANED`). Arbitrary system temp files are untouched.
- **Cross-Machine Portable Key Wrapping**:
  - Backup exports accept a user-supplied recovery passphrase.
  - Key Derivation: `scrypt` (N=32768, r=8, p=1, 32-byte salt) derives a 256-bit wrapping key.
  - The local evidence master key is wrapped via AES-256-GCM and stored inside the encrypted backup archive.
  - On a clean ExamGuard node, the backup package and recovery passphrase can unwrap the evidence master key and reprotect it under the destination node's DPAPI profile without requiring the original Windows profile.
- **Manifest Privacy**:
  - Outer `backup-manifest.json` exposes strictly non-sensitive cryptographic headers: backup format version (`BACKUP_FORMAT_V2_ENCRYPTED`), scrypt KDF salt/parameters, payload filenames, and ciphertext SHA-256 hashes.
  - All sensitive session names, candidate identifiers, and event counts are encapsulated inside the encrypted payload.
- **Memory Hygiene & Accurate Cleanup Semantics**:
  - Sensitive plaintext buffers (passphrases, derived keys, serialized database bytes) are released and mutable bytearrays overwritten on a best-effort basis.
  - In accordance with truthful security reporting, no claim of "guaranteed RAM wiping" or "forensic erase" is made, recognizing Python runtime allocator behavior, memory garbage collection, and OS paging.
- **Operational Safety Boundary**:
  - `AUTOMATIC_RESTORE = NO`. The system validates cryptographic unwrap capability and database integrity in memory, but automatic destructive live-restore remains disabled in this foundation pass.

---

## 6. Multi-Camera Single-Node Orchestration

### 6.1 Multi-Camera Safety Principles
- **No Cross-Camera Biological Identification**: The system explicitly does **not** perform facial recognition, appearance Re-ID, or automatic candidate linking across cameras.
- **Scoped Identity**: Every track is uniquely identified by `(camera_id, track_id)`.
- **Model Sharing**: A singleton `ModelRegistry` hosts shared YOLOv8/v26 detection, posture, and head-pose models to prevent Out-Of-Memory on 4 GB GPUs (NVIDIA RTX 3050).
- **GPU Scheduling**: Fair inference dispatch with bounded per-camera queues (`DROP_STALE_ON_BACKPRESSURE`) ensures camera 1 never starves camera 2.
- **State Isolation**: Temporal buffers, standing baselines, phone associators, and ring buffers are completely isolated per `camera_id`.
- **Failure Resilience**: A lost frame signal or disconnection on camera B triggers auto-reconnect with exponential backoff without stalling camera A.

---

## 7. Audit Logging & Review Non-Repudiation

### 7.1 Tamper-Evident Hash Chain
Each audit log entry computes:
$$\text{entry\_hash} = \text{SHA256}(\text{previous\_hash} + \text{audit\_id} + \text{session\_id} + \text{action} + \text{actor\_id} + \text{details\_json} + \text{created\_at})$$
Historical log modification invalidates downstream hashes, revealing tampering during system audits.

### 7.2 Proctor Attribution
Review actions (CONFIRM / DISMISS) derive `reviewer_id` and `reviewer_name` directly from the authenticated server session, preventing client-side spoofing.
