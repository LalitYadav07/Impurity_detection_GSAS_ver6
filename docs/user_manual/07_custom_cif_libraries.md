# Custom CIF Libraries

Custom CIF libraries let users bring their own candidate structures. This is important when the built-in MP/COD catalog does not contain project-specific materials, unpublished structures, substituted compounds, or instrument-specific reference phases.

## Library Modes

| Mode | Behavior |
| --- | --- |
| Built-in catalog + my CIFs | The search includes both the built-in catalog and uploaded CIFs. |
| Only my CIFs | The search is restricted to the uploaded CIF collection. |

## Recommended Naming

Use names that explain the source and purpose:

- `battery_cathodes_2026`
- `beamline_reference_cans`
- `project_x_oxide_variants`
- `docs_demo_custom_small`

Avoid names like `test`, `new`, or `final`.

## Build Workflow

1. Open a persistent workspace.
2. Open Candidate Library.
3. Expand Create Library from CIFs.
4. Choose library type.
5. Enter a library name.
6. Upload CIF files.
7. Build the database pack.
8. Confirm that the new library is selected or available.

## Scientific Notes

Custom libraries are powerful but can also bias the search. Use **Only my CIFs** only when you deliberately want to exclude all other phases. Use **Built-in + my CIFs** when you want broad safety plus project-specific additions.

## Troubleshooting

If a custom library fails:

- inspect the CIF parse error,
- remove malformed CIFs and rebuild,
- avoid duplicate names,
- check whether the CIF has valid cell and symmetry fields,
- try a small subset first.

