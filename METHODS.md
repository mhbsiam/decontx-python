# Methods and comparison

## Why use DecontX?

Ambient RNA contamination occurs when droplets capture mRNA from lysed or stressed cells along with other cells. This causes:
- Cross-contamination between cell types
- Blurred cell type boundaries  
- False positive marker gene expression
- Reduced clustering quality

DecontX models each cell as a mixture of two parts:
1. **Native transcripts** from the cell's true type
2. **Contaminating transcripts** from other cell types in the sample

## Method comparison

We ran a benchmarking study. The table below shows the results.

| Method | Ambient RNA Removed | Precision | Conservativeness |
|--------|-------------------|-----------|------------------|
| **SoupX** | ~65% | High | Very conservative |
| **DecontX** | ~90% | Medium-High | Balanced |
| **CellBender** | ~90% | Medium | More aggressive |

**Recommendation**:
- Use **SoupX** when you want maximum safety and the fewest false positives.
- Use **DecontX** for balanced contamination removal in standard workflows.
- Use **CellBender** when you can replace your entire preprocessing pipeline.
