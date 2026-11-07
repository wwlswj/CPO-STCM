# Publish commands

Run these only after explicit authorization and successful GitHub/Zenodo
authentication.

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

Do not create the final data tag or Release until the Zenodo DOI and final
metadata review are complete.

## Zenodo

1. Authenticate to Zenodo with the account that should own both records.
2. Create the software deposit from the tagged GHCR/source release.
3. Upload the full CPO-LED Bench v2.0.0 archive and corrected v2.0.1 metadata
   as a separate dataset deposit.
4. Mint version and concept DOIs.
5. Update `CITATION.cff`, the Software Impacts manuscript, and the dataset
   citation with the final DOI values.
