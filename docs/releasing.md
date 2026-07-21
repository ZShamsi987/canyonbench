# Release checklist

1. Freeze code and data schema versions and update both changelogs.
2. Confirm every public frame has a valid source/license record and contains no identifiable people.
3. Run annotation agreement and record adjudication decisions.
4. Recompute masks, coverage, grids, registration residuals, and weak-label comparisons from frozen inputs.
5. Run `canyonbench validate-release RELEASE_DIR` with image files present.
6. Run all code checks in `scripts/check.sh` and score a fixture model end to end.
7. Create content hashes, freeze geographic splits, and confirm no block or segment leakage.
8. Tag matching code and data versions. Cite the exact code commit in the dataset card.
9. Upload the dataset to Hugging Face and archive the frozen release on Zenodo for a DOI.
10. Replace DOI/paper placeholders in both citation files and verify cross-links.

GitHub is for source, schemas, and small fixtures. Use Hugging Face or the DOI archive for imagery; Git LFS alone is not the preservation record.

