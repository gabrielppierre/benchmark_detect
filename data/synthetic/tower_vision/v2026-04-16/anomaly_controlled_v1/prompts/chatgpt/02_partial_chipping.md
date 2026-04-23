You are editing a real inspection photo crop of a power-line insulator.

Task:
Insert exactly one realistic visual anomaly into the insulator only.

Primary goal:
Create a photorealistic edited version of the input image that still looks like a real inspection crop, while adding one localized anomaly of type partial chipping with severity moderate.

Hard constraints:
- Keep the same framing, aspect ratio, resolution, perspective, and camera viewpoint.
- Preserve the same background, lighting, shadows, color balance, and overall photographic style.
- Preserve the global geometry and identity of the insulator.
- Edit only a small localized region of the insulator.
- Do not add any new objects outside the insulator.
- Do not change the tower, cables, sky, background, or scene composition.
- Do not create multiple defects.
- Do not turn the image into an illustration, CGI render, painting, or synthetic-looking artwork.
- Do not degrade the whole image.
- The result must remain visually plausible for a technical inspection scenario.

Anomaly specification:
- anomaly scope: insulator only
- anomaly type: partial chipping
- severity: moderate

Desired behavior:
- The anomaly must be clearly visible but still plausible.
- The anomaly must affect only one limited area of the insulator.
- The rest of the insulator should remain intact.
- Material appearance must remain coherent with the apparent insulator material in the image.
- Keep texture realism and local consistency.

Negative constraints:
Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, lighting setup, or overall scene composition. Do not add text, watermark, logo, extra objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or global image degradation.

Output:
Return one edited image only.
