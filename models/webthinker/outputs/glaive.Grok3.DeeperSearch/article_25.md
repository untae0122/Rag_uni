

### Key Points
- Research suggests parallel transmission in MRI improves spatial encoding and image fidelity by controlling RF fields, especially with multi-channel transmit coils.
- The Bloch-Siegert shift seems likely to enhance B1 mapping, aiding accurate RF pulse design in these systems.
- Historical work by Richard Ernst and Raymond Damadian likely laid groundwork, though not directly on parallel transmission, through NMR and MRI advancements.

---

### Introduction to Parallel Transmission in MRI
Parallel transmission techniques, using multi-channel transmit coils, allow for better shaping of the radiofrequency (RF) field in MRI systems. This is particularly important in modern high-field MRI, where field inhomogeneities can distort images. By controlling multiple RF channels, these systems can improve spatial encoding, ensuring the MRI machine accurately maps where signals come from, and enhance image fidelity, leading to clearer and more reliable images.

### Impact on Spatial Encoding and Image Fidelity
Parallel transmission helps mitigate B1 inhomogeneity, a common issue at high fields, by tailoring RF pulses to achieve uniform excitation. This improves spatial encoding, as the system can better distinguish different locations in the body. The Bloch-Siegert shift, a method to map the RF field (B1 mapping), is crucial here. It ensures the RF pulses are accurately calibrated, which research suggests leads to higher fidelity in reconstructed images, reducing artifacts and improving contrast.

An unexpected detail is that the Bloch-Siegert shift can also enable selective excitation without traditional gradients, offering new ways to encode spatial information, potentially reducing noise and patient discomfort from gradient switching.

### Historical Antecedents
While Richard Ernst and Raymond Damadian did not work directly on parallel transmission, their contributions were foundational. Ernst's work on NMR spectroscopy, including Fourier transform methods and pulse sequences, influenced complex RF pulse designs used today. Damadian's development of the first whole-body MRI scanner addressed early RF uniformity challenges, indirectly foreshadowing parallel transmission's focus on field control.

---

---

### Survey Note: Detailed Analysis of Parallel Transmission and Historical Context in MRI

This section provides a comprehensive exploration of how parallel transmission techniques, particularly with multi-channel transmit coils, affect spatial encoding and image fidelity in the context of the Bloch-Siegert shift, alongside historical antecedents from Richard Ernst and Raymond Damadian. The analysis is grounded in recent research and historical developments, aiming to offer a thorough understanding for both technical and lay audiences.

#### Background on Parallel Transmission in MRI
Magnetic Resonance Imaging (MRI) relies on magnetic fields and radio waves to generate detailed images of the body's internal structures. Spatial encoding, the process of determining signal origins, typically uses gradient magnetic fields to create spatial variations in the magnetic field. However, at ultrahigh fields (e.g., 7T and above), RF field (B1) inhomogeneities can lead to non-uniform excitation, affecting image quality. Parallel transmission, introduced around 2003 by Katscher et al. in their seminal work "Transmit SENSE" ([Transmit Sense](https://www.researchgate.net/publication/10966280_Transmit_Sense)), addresses this by using multiple transmit coils driven independently. This allows for spatial and temporal control of the RF field, improving excitation uniformity and reducing specific absorption rate (SAR), a measure of RF energy absorbed by the body.

The technique builds on parallel imaging, which used multiple receive coils for faster data acquisition, as seen in works like SENSE by Pruessmann et al. (1999) and SMASH by Sodickson and Manning (1997). Parallel transmission extends this concept to transmission, enabling multidimensional, spatially selective RF pulses, which is particularly vital for high-field MRI where wavelength effects in the body cause field distortions.

#### Impact on Spatial Encoding
Spatial encoding in MRI traditionally involves B0 gradients for slice selection, phase encoding, and frequency encoding. Parallel transmission enhances this by allowing precise RF field shaping. Each transmit coil can contribute to the overall field, and by adjusting amplitude and phase, the system can create tailored excitation patterns. This is crucial for maintaining accurate spatial encoding, as non-uniform RF fields can lead to variable flip angles, distorting spatial information.

The Bloch-Siegert shift, a frequency shift caused by off-resonance RF pulses, plays a key role in B1 mapping, as detailed in "B1 Mapping by Bloch-Siegert Shift" ([B1 Mapping by Bloch-Siegert Shift](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2933656/)). This method encodes B1 information into signal phase, offering advantages in speed and accuracy. In parallel transmission, accurate B1 mapping ensures RF pulses are designed to compensate for inhomogeneities, thus improving spatial encoding. For instance, research in "Improved Bloch-Siegert Based B1 Mapping by Reducing Off-Resonance Shift" ([Improved Bloch-Siegert Based B1 Mapping by Reducing Off-Resonance Shift](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3669656/)) shows how this enhances sensitivity, reducing errors in high-field MRI.

An innovative application is RF gradient-encoded MRI, where the Bloch-Siegert shift enables B0-gradient-free selective excitation, as seen in "Selective excitation localized by the Bloch-Siegert shift and a B1 gradient" ([Selective excitation localized by the Bloch-Siegert shift and a B1 gradient](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9246893/)). This method uses an off-resonant pulse and a frequency-selective pulse, designed via the Shinnar-Le Roux algorithm, to achieve spatial selectivity. Experimental results at 47.5 mT showed robust performance, with parameters like off-resonance shifts (-0.033 mT/kHz to -0.036 mT/kHz) and minimal RF amplifier distortion, suggesting potential for new imaging approaches without traditional gradients.

| Parameter                  | Value/Detail                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| MRI Scanner Field Strength | 47.5 mT                                                                     |
| \( \omega_{\text{off}} \) Examples | 2.5 kHz, 5.0 kHz, 10.0 kHz, 7.5 kHz, 16.37 kHz, 9.485 kHz, 19.25 kHz, 8.17 kHz |
| Off-Resonance Shift        | -0.033 mT/kHz, -0.019 mT/kHz, -0.038 mT/kHz, -0.036 mT/kHz                  |
| RF Amplifier Droop Effects | RFSE: -0.5 dB (~13% reduction), -1.5 dB (complete degradation); BSSE: 0.004 mT (-0.5 dB), 0.013 mT (-1.5 dB) |
| Experimental Pulse Details | 90° flip, PBC = 0.14 mT, PBW = 0.03 mT, \( \omega_{\text{off}} = 7.5 \, \text{kHz} \), \( T = 6.22 \, \text{ms} \), \( T_{\text{ex}} B = 4 \) |
| Multiphoton Resonance (N=3)| \( |M_{xy}|/M_0 = 0.075 \) (90°), \( M_z/M_0 = 0.918 \) (180°)               |
| Code URL                   | https://github.com/jonbmartin/sigpy-rf                                       |

This table summarizes key experimental parameters, highlighting the practical implementation of the Bloch-Siegert shift in selective excitation, which can enhance spatial encoding in parallel transmission systems.

#### Impact on Image Fidelity
Image fidelity, or the accuracy and clarity of reconstructed images, is directly tied to uniform excitation and minimal artifacts. Parallel transmission improves fidelity by reducing B1 inhomogeneity, as noted in "Parallel Transmission for Ultrahigh Field MRI" ([Parallel Transmission for Ultrahigh Field MRI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7039313/)). By designing RF pulses that account for individual coil contributions, it minimizes contrast variations, enhancing diagnostic reliability. The Bloch-Siegert shift's role in B1 mapping ensures these pulses are calibrated, reducing errors that could degrade image quality.

An interesting finding is the potential for parallel transmission to enable simultaneous multislice imaging and inner volume imaging, as reviewed in the same paper, which could further improve fidelity by optimizing coverage and resolution. This is particularly relevant at ultrahigh fields, where traditional methods struggle with SAR and field uniformity.

#### Historical Antecedents: Contributions of Ernst and Damadian
The question of historical antecedents leads us to Richard Ernst and Raymond Damadian, pioneers in NMR and MRI. Ernst, awarded the Nobel Prize in 1991 for Fourier transform NMR, developed techniques like pulse sequences and phase cycling, which are foundational for modern RF pulse design. While his work, detailed in NMR spectroscopy texts, did not directly address parallel transmission, it influenced the complex pulse designs used today, as seen in the Shinnar-Le Roux algorithm's roots in NMR.

Damadian, credited with the first whole-body MRI scanner in the 1970s, focused on practical imaging challenges, including RF field uniformity, as discussed in early MRI literature. His work, while not involving multiple transmit coils, addressed issues that parallel transmission later tackled, such as ensuring consistent excitation across large volumes.

Parallel transmission itself emerged in 2003 with Katscher et al.'s "Transmit SENSE," building on parallel imaging from the late 1990s (e.g., SENSE, SMASH). Thus, while Ernst and Damadian did not directly foreshadow parallel transmission, their contributions to NMR and MRI provided the theoretical and practical groundwork, particularly in RF pulse design and field uniformity, indirectly paving the way for these advancements.

#### Conclusion
Parallel transmission, leveraging multi-channel transmit coils, significantly enhances spatial encoding and image fidelity in MRI, with the Bloch-Siegert shift playing a critical role in B1 mapping and enabling innovative encoding methods. Historical work by Ernst and Damadian, though not directly on parallel transmission, laid essential foundations through NMR spectroscopy and early MRI development, influencing the field's evolution toward advanced RF techniques.

---

### Key Citations
- [Transmit Sense](https://www.researchgate.net/publication/10966280_Transmit_Sense)
- [B1 Mapping by Bloch-Siegert Shift](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2933656/)
- [Improved Bloch-Siegert Based B1 Mapping by Reducing Off-Resonance Shift](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3669656/)
- [Selective excitation localized by the Bloch-Siegert shift and a B1 gradient](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9246893/)
- [Parallel Transmission for Ultrahigh Field MRI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7039313/)