# MMScan VG/QA Benchmark Samples Generation

Here we mainly provide sample codes to show how we derive the benchmark samples from meta-annotations. Since the manual check mainly relies on human annotations and the UI design, we have shown the UI screenshot in the attached rebuttal pdf, which is more clear to understand the process.

## Benchmark Samples Derivation

We provide the code for samples derivation pipeline: GPT-based property extraction --> template-based generation --> GPT-based refinement. After these generation, we verify the logic of derived samples and ensure the overall quality following the random inspection and 95% principle.

Among the three steps, the last one simply prompts GPT to refine all the samples to enhance its diversity and clarity in the presentation, so we mainly show the code for the first two steps.

1. GPT-based property extraction

   We first use GPT to extract knowledge from meta-annotations that is used in the QA/VG samples generation,mainly including: (1) Unique descriptions, which depict the unique attribute of an object; (2) Common descriptions, which depict common attributes of objects; (3) Differences between two objects in object pairs, which depict different attributes of objects.
   ```bash
   srun python -m anno_gpt_extraction.main # need multi-process
   ```

2. Sample Generation

   Next, we generate VG/QA samples based on meta-annoations, GPT-based extracted properties and templated rules.

   ```bash
   python -m samples_generation.VG_generate
   python -m samples_generation.QA_generate
   ```