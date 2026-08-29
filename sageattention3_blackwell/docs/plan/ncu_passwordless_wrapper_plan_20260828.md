# Restricted Passwordless NCU Wrapper Plan

## Goal

Profile the SageAttention v3 kernel without rebooting and without entering a
sudo password on each run.

## Approach

Install a root-owned helper at
`/usr/local/sbin/sageattention3-ncu-attach` that accepts no arguments and can
only attach Nsight Compute to localhost port 49220 with the fixed profiling
configuration. Add one sudoers rule allowing only `yiliu7` to run that helper
without a password.

The existing launcher will:

1. Start the Python profiling target as `yiliu7`.
2. Invoke the fixed root helper through passwordless sudo.
3. Copy the completed report from a root-controlled directory into
   `~/.local/state/sageattention3/`.

## Security Decisions

- Do not grant passwordless access to arbitrary `ncu` commands because NCU can
  launch programs as root.
- Keep the root helper argument-free and root-owned.
- Write the privileged report under `/var/lib/sageattention3-ncu`, not a
  user-writable path, to prevent symlink-based privileged file replacement.
- Require one authenticated sudo run to install the helper and sudoers rule.

## Validation

- Validate the sudoers fragment with `visudo -cf` before installation.
- Confirm the helper rejects arguments.
- Confirm the user launcher connects through launch/attach mode.
- Confirm the resulting report exists and is readable by `yiliu7`.
- Confirm the launcher terminates its waiting target if attachment fails.

## Rollback

Remove these two fixed files with sudo:

- `/usr/local/sbin/sageattention3-ncu-attach`
- `/etc/sudoers.d/sageattention3-ncu`
