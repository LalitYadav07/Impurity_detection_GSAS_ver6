# RADAR-PD workflow chaos and UX audit — 2026-11-09

Source: `codex/chaos-ux-workflows`, based on the v0.6.63 frontend. The isolated local QA app at `http://127.0.0.1:8129` used this source and the existing test Galaxy History. This was not a deployment. The walkthrough restored published HB-2A work and did not submit a new analysis, recipe, or model campaign.

## Most consequential finding

Changing the editable Live beamline after restoring a stopped watch erased its displayed controller and results. A saved POWGEN WebMON scope could also be invalidated by edits to the draft IPTS. The source now keeps the displayed watch immutable until explicit workspace clear; the dashboard and setup rail name the displayed watch and separate draft. Clear explicitly retires the watch and WebMON state, including during delayed reads. Regression tests cover this lifecycle.

## Workflow coverage

`Browser` means an interactive walkthrough in the isolated QA app. `Fixture` means deterministic automated scenarios; it is not a claim that a new facility job ran. `Archive` means read-only directory discovery on the DGX.

| Guide task | Coverage and observed result | Remaining external gate |
| --- | --- | --- |
| 1. Open workspace | Browser startup and beamline choice; fixture launch/session tests | NDIP deployment of this branch |
| 2. Guided Live setup | Browser switched POWGEN → NOMAD → HB-2A → SNAP and guided → manual → guided; selections persisted; fixture assistant language/inventory tests | Live discovery against a new active experiment |
| 3. Inputs and profiles | Browser saw beamline-specific profile/normalization controls; fixture upload/library/duplicate/profile tests; inline History refresh and wrong-slot guidance added | Actual wrong-slot Galaxy upload in a disposable History |
| 4. Check and one-scan test | Fixture preflight/profile/scope guards; clickable missing-input chips added | No one-scan Galaxy job submitted |
| 5. Monitor series | Fixture watch, queue, stop and resume tests; session/job distinction in existing confirmation copy | No new continuous monitor started |
| 6. Inspect a fit | Browser opened published HB-2A scan 33; observed/calculated/residual and Bragg tick plot rendered; scientific qualifications visible | Independent calibration review |
| 7. Trends | Browser loaded four HB-2A scans and scan-number trend; fixture condition and result tests | No new reviewed temperature mapping |
| 8. Transitions | Browser loaded 4 scans, 3 comparisons, 35 U-Net candidate regions; heat map and linked plots rendered; fixture bank/coverage tests | Scientific feature acceptance remains user review |
| 9. Recipe | Browser inspected one saved GPX and saw actionable approval prerequisite; fixture recipe/transport/stage tests | No recipe approved or polishing job submitted |
| 10. Refined results | Fixture result/provenance/transport tests; staged outputs not opened in browser | A completed new recipe job |
| 11. Model exploration | Browser confirmed four exact starting GPX options; fixture symmetry/unresolved/desktop tests; campaign prerequisites made actionable | No magnetic/structural campaign submitted |
| 12. Live acquisition | Fixture WebMON scope, stale, pinning, revision and stream tests; status chips added | No live WebMON producer snapshot exercised in browser |
| 13. Recover | Browser selected a published HB-2A checkpoint, got an automatic preview, restored it stopped, then edited draft beamline without losing four results; fixture lifecycle tests | Other Histories/checkpoint vintages |
| 14. Contact developer | Fixture feedback tests from the broad suite | No feedback message sent |
| 15. Save/clear/end | Fixture clear/end/session/late-read tests; browser retained the restored dashboard | No real NDIP session ended |

Read-only archive probes found mixed product families under POWGEN `IPTS-2767`, NOMAD `IPTS-38746`, HB-2A `IPTS-28749`, and SNAP `IPTS-33088`. These probes established realistic discovery inputs; they did not validate scientific fits or modify the archives.

## UX changes

- History checkpoint selection previews identity and restrictions immediately. Restore remains the explicit commit.
- Results and transition/refinement views collapse the setup rail. Narrow layouts use a drawer, scrim, and sticky close control; Guided Live stays in front of the scrim.
- The setup rail shows `DISPLAYED WATCH` separately from the editable draft. The dashboard shows `Restored watch · stopped` and `Setup draft differs`; WebMON ownership uses the saved watch.
- Missing-input chips lead to the relevant setup fields for monitoring, recipe approval and model campaigns. History refresh is adjacent to Galaxy input selectors.
- WebMON shows following/pinned, automatic analysis and stale-revision status. Recipe approval and future enrollment have visible status chips.
- Dense publication and scientific warnings in the full result are grouped under concise, expandable summaries; the scientific-review status stays visible.
- The bundled 15-task guide was updated for the new controls and verified.

## Validation and limits

The broad suite initially completed with **4,319 passed, 11 skipped and 3 failed**. Those failures were confined to WebMON lifecycle expectations after the scope change. The corrected lifecycle and neighboring app tests then passed **231/231**. A separate result/navigation/transition/refinement suite passed **507/507**, and the UI contract suite passed **16/16** after updating its expected publication wording. Two optional tests could not collect in the host test environment because `Cheetah` is absent. `git diff --check`, Python compilation and the bundled guide content check passed. Browser screenshots in this directory show the restored-watch status, HB-2A fit, and final qualification disclosure.

This audit covers all 15 guide tasks through a mix of browser, fixture and archive checks. It does **not** certify a new end-to-end scientific execution for each beamline. High-cost or external actions (new Galaxy analyses, live WebMON capture, model campaigns, feedback sending, and a hosted GSAS-II desktop switch) remain explicit integration gates.
