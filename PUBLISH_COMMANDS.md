# Publish commands

These commands record the completed public-release workflow. Do not rerun tag
or Release creation commands against existing immutable assets.

## GitHub RC

```powershell
gh auth login -h github.com
git status --short
git push -u origin main
git push origin v1.0.0-rc.1
gh release create v1.0.0-rc.1 `
  .\dist\cpo_stcm-1.0.0rc1-py3-none-any.whl `
  .\dist\cpo_stcm-1.0.0rc1.tar.gz `
  .\dist\CPO-STCM-v1.0.0-rc.1-source.zip `
  .\dist\CPO-STCM-v1.0.0-rc.1-release-notes.md `
  .\dist\SHA256SUMS.txt `
  --repo wwlswj/CPO-STCM `
  --title "CPO-STCM v1.0.0-rc.1" `
  --notes-file .\RELEASE_NOTES_v1.0.0-rc.1.md
```

If the GitHub repository does not yet exist, create it first with an
authenticated `gh` session:

```powershell
gh repo create wwlswj/CPO-STCM --public --source . --push
```

Then push the annotated tag and create the Release.

## Data metadata review branch

From the local CPO-LED Bench clone:

```powershell
git push origin metadata/v2.0.1
```

The dataset Zenodo DOI and metadata review were completed on 2026-11-08. Create
the `v2.0.1` tag and Release only after the metadata PR is merged.

## Zenodo completed steps

1. Authenticated to Zenodo with the record owner account.
2. Created and published the software record from the RC release.
3. Uploaded the full CPO-LED Bench v2.0.0 archive and corrected v2.0.1 metadata
   as a separate dataset record.
4. Registered version and concept DOIs for both records.
5. Updated citation metadata, the Software Impacts manuscript, and release
   documentation with the final DOI values.
