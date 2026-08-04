# RADAR-PD Hosted User Manual

This folder contains the editable source and generated deliverables for the
hosted RADAR-PD user manual.

Source of truth:

- Markdown chapters in this folder.
- Screenshot inventory in `screenshot_manifest.yaml`.
- Static assets in `screenshots/`.

Generated outputs:

- `dist/radar_pd_hosted_user_manual.html`
- `dist/radar_pd_hosted_user_manual.pdf`
- `dist/radar_pd_hosted_user_manual.md`

Build commands:

```powershell
python scripts/build_user_manual.py --check
python scripts/build_user_manual.py --html
python scripts/build_user_manual.py --pdf
python scripts/build_user_manual.py --all
```

Screenshot policy:

- Use the hosted app at `http://128.219.184.26:8502/`.
- Use the clean documentation workspace `docs_demo` with PIN `2468`.
- Do not use private `u1` screenshots in the public manual.
- Save screenshots with stable filenames listed in `screenshot_manifest.yaml`.
- If a screenshot is referenced by a Markdown image link, it must exist before
  `--check` will pass.

Manual scope:

- Web-hosted RADAR-PD app usage.
- API usage for submitted jobs and result download.
- Scientific interpretation of options and outputs.
- No server deployment, VM administration, or developer setup instructions.

