

### Key Points
- It seems likely that using virtual machines (VMs) with isolated networks can effectively integrate hands-on labs into your lesson plan for threat modeling and vulnerability assessment.
- Research suggests providing pre-configured VMs with necessary tools, like OWASP Threat Dragon for threat modeling and OWASP ZAP for vulnerability assessment, ensures consistency and compliance with security policies.
- The evidence leans toward setting up labs with a target VM (e.g., Metasploitable) and an attacker VM on an internal network to maintain a controlled environment, aligning with strict network segmentation.

### Setting Up Virtual Machines
To integrate hands-on labs, use virtualization software like [VirtualBox](https://www.virtualbox.org/) or [VMware Workstation Player](https://www.vmware.com/products/workstation-player.html). Set up two VMs: one as an "attacker" with tools for modeling and assessment, and another as a "target" with vulnerable systems like Metasploitable or DVWA. This setup allows students to practice safely without risking the institution's network.

### Ensuring a Controlled Environment
Configure the VMs on an internal or host-only network to isolate them from the main network, adhering to your institution's strict security policies. Provide pre-configured VM images to students to ensure consistency, and include a lab manual with clear instructions. Emphasize ethical practices to prevent unauthorized actions outside the lab.

### Lesson Plan Integration
Structure your lesson plan with theoretical introductions to threat modeling and vulnerability assessment, followed by hands-on labs. For example, students can model threats using OWASP Threat Dragon on the attacker VM, then assess vulnerabilities on the target VM using OWASP ZAP, linking both activities to reinforce learning.

---

### Survey Note: Detailed Implementation for Hands-On Labs in Threat Modeling and Vulnerability Assessment

This note provides a comprehensive guide for integrating hands-on labs using virtual machines (VMs) into a lesson plan focused on threat modeling and vulnerability assessment, while ensuring compliance with institutional network segmentation and security policies. The approach is designed to offer practical, replicable exercises in a controlled environment, suitable for educational settings as of March 26, 2025.

#### Background and Rationale
Threat modeling involves identifying and prioritizing potential threats to a system, while vulnerability assessment focuses on finding and evaluating weaknesses. Hands-on labs using VMs are ideal for teaching these concepts, as they create isolated environments for students to apply theoretical knowledge without risking real-world systems. Given the strict network policies mentioned, isolation is crucial to prevent any lab activities from impacting the institution's network.

#### Virtualization Platform Selection
The choice of virtualization software is critical for setting up labs. [VirtualBox](https://www.virtualbox.org/) and [VMware Workstation Player](https://www.vmware.com/products/workstation-player.html) are recommended due to their widespread use, free availability for educational purposes, and support for network isolation. These platforms allow for the creation of multiple VMs on a single host, which is practical for classroom or individual student use.

#### VM Configuration and Tools
To facilitate hands-on learning, configure at least two VMs:
- **Attacker VM:** Install a Linux distribution such as Kali Linux, which comes with pre-installed security tools. Add threat modeling tools like [OWASP Threat Dragon](https://owasp.org/www-project-threat-dragon/) (open-source) and Microsoft's Threat Modeling Tool (free, available at [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=49168)). For vulnerability assessment, include tools like [OWASP ZAP](https://www.zaproxy.org/) (open-source), [OpenVAS](https://www.openvas.org/) (open-source), and Nessus Essentials (free for non-commercial use).
- **Target VM:** Use pre-built vulnerable systems such as Metasploitable (downloadable at [Rapid7](https://information.rapid7.com/download-metasploitable-2017.html)) or DVWA ([DVWA Website](http://www.dvwa.co.uk/)). These systems are designed for educational purposes and contain known vulnerabilities for students to assess.

For a more complex setup, consider creating a small virtual network with multiple target VMs (e.g., web server, database server) to simulate real-world architectures, enhancing the realism of the labs.

#### Network Segmentation and Security Compliance
Given the institution's strict network segmentation and security policies, configure the VMs on an internal or host-only network within the virtualization software. This ensures that the VMs can communicate with each other for lab exercises but are isolated from the institution's main network and the internet, mitigating risks of unauthorized access or data leakage.

If internet access is required (e.g., for downloading updates), consider setting up a separate VM as a gateway with controlled outbound traffic, or instruct students to download necessary files on their host machine and transfer them via shared folders. This approach maintains isolation while meeting practical needs.

To ensure compliance, consult with the institution's IT department for specific guidelines, but general best practices include:
- Disabling bridged or NAT networking that connects to the institution's network.
- Regularly updating and patching the VMs to minimize vulnerabilities within the lab environment.
- Including a lab safety section in the curriculum, emphasizing ethical hacking practices and prohibiting unauthorized testing outside the lab.

#### Lesson Plan Development
The lesson plan should integrate theoretical learning with practical labs, structured as follows:

1. **Introduction to Threat Modeling (1 Hour):**
   - Cover definitions, importance, and methodologies like STRIDE, PASTA, and Attack Trees.
   - Discuss tools like OWASP Threat Dragon and Microsoft's Threat Modeling Tool, highlighting their use in identifying threats.

2. **Hands-On Lab: Threat Modeling (2 Hours):**
   - Provide students with the architecture diagram of the target VM (e.g., a web application on DVWA).
   - Instruct them to use the attacker VM's threat modeling tool to identify potential threats, document findings, and propose mitigations. For example, using STRIDE, they might identify risks like SQL injection (information disclosure) or denial-of-service attacks.

3. **Introduction to Vulnerability Assessment (1 Hour):**
   - Explain types of vulnerabilities (e.g., SQL injection, cross-site scripting) and assessment techniques.
   - Introduce tools like OWASP ZAP for web application scanning and OpenVAS for network vulnerability assessment.

4. **Hands-On Lab: Vulnerability Assessment (2 Hours):**
   - Have students use the attacker VM to scan the target VM for vulnerabilities, interpret scan results, and prioritize findings based on risk.
   - Encourage comparison with threats identified in the threat modeling lab to see correlations, enhancing understanding of the relationship between the two processes.

5. **Integration and Discussion (1 Hour):**
   - Facilitate a discussion on how threat modeling can guide vulnerability assessment, and vice versa.
   - Reflect on lab experiences, addressing any challenges and reinforcing key concepts. Encourage students to share unexpected findings, such as vulnerabilities not initially modeled.

#### Ensuring Replicability
To ensure students can replicate exercises, provide pre-configured VM images with all tools and target systems installed. Include a detailed lab manual with step-by-step instructions, covering:
- How to import and run the VMs in VirtualBox or VMware.
- Network configuration settings for isolation.
- Specific commands or procedures for using each tool (e.g., launching OWASP ZAP, initiating a scan).
- Expected outcomes and how to document results.

Specify minimum hardware requirements for students' laptops, such as 8GB RAM and a dual-core CPU, to ensure smooth operation of multiple VMs. This approach minimizes setup time and ensures consistency across all student environments.

#### Additional Considerations
- **Ethics and Safety:** Include a module on cybersecurity ethics, emphasizing that lab techniques should only be applied in controlled environments with permission. This is crucial to prevent misuse and align with institutional policies.
- **Scalability and Resources:** Plan for scalability by ensuring the virtualization software can handle multiple VMs if needed for group projects. Consider cloud-based lab solutions (e.g., [SecureFlag](https://blog.secureflag.com/2022/11/04/threat-modelling-labs-now-available-on-secureflag/)) if institutional infrastructure allows, though local setups are likely preferred for isolation.
- **Licensing:** Use open-source or free educational licenses for all tools to avoid cost barriers, such as OWASP ZAP and OpenVAS, which are freely available.

#### Unexpected Detail: Linking Theory to Practice
An unexpected benefit of this integrated approach is how it bridges theoretical threat modeling with practical vulnerability assessment, allowing students to see real-world implications. For instance, identifying a threat like "unauthorized data access" in modeling, then finding an SQL injection vulnerability in assessment, reinforces the importance of proactive security design.

#### Table: Recommended Tools and Resources

| **Category**               | **Tool**                     | **Description**                                      | **URL**                                                                 |
|----------------------------|------------------------------|------------------------------------------------------|-------------------------------------------------------------------------|
| Threat Modeling Tool       | OWASP Threat Dragon          | Open-source tool for diagramming and modeling threats | [OWASP Threat Dragon](https://owasp.org/www-project-threat-dragon/)     |
| Threat Modeling Tool       | Microsoft Threat Modeling Tool | Free tool for threat modeling, supports STRIDE       | [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=49168) |
| Vulnerability Assessment   | OWASP ZAP                    | Open-source web application scanner                 | [OWASP ZAP](https://www.zaproxy.org/)                                  |
| Vulnerability Assessment   | OpenVAS                      | Open-source network vulnerability scanner            | [OpenVAS](https://www.openvas.org/)                                    |
| Vulnerable Target System   | Metasploitable               | Pre-built vulnerable Linux system for practice       | [Rapid7](https://information.rapid7.com/download-metasploitable-2017.html) |
| Vulnerable Target System   | DVWA                         | Damn Vulnerable Web Application for web security     | [DVWA Website](http://www.dvwa.co.uk/)                                 |

#### Conclusion
By setting up isolated VMs with pre-installed tools, structuring a comprehensive lesson plan, and ensuring compliance with security policies, you can create an effective and secure learning environment. This approach not only meets educational goals but also prepares students for real-world cybersecurity challenges, aligning with the institution's need for a controlled and segmented network.

### Key Citations
- [HandsOn Training Practical Threat Modeling](https://www.handson-training.com/page/Practical-Threat-Modeling)
- [O'Reilly Hands-on Threat Modeling](https://www.oreilly.com/live-events/hands-on-threat-modeling/0636920340430/)
- [Center for Cyber Security Training Hands-On Threat Modeling](https://ccsecuritytraining.com/training/hands-on-threat-modeling/)
- [Practical DevSecOps Certified Threat Modeling Professional](https://www.practical-devsecops.com/certified-threat-modeling-professional/)
- [NICCS Hands-On Threat Modeling](https://niccs.cisa.gov/education-training/catalog/center-cyber-security-training-llc/hands-threat-modeling)
- [OWASP Threat Dragon Website](https://owasp.org/www-project-threat-dragon/)
- [Microsoft Threat Modeling Tool Download](https://www.microsoft.com/en-us/download/details.aspx?id=49168)
- [OWASP ZAP Website](https://www.zaproxy.org/)
- [OpenVAS Website](https://www.openvas.org/)
- [Rapid7 Metasploitable Download](https://information.rapid7.com/download-metasploitable-2017.html)
- [DVWA Website](http://www.dvwa.co.uk/)
- [VirtualBox Website](https://www.virtualbox.org/)
- [VMware Workstation Player Website](https://www.vmware.com/products/workstation-player.html)
- [SecureFlag Threat Modelling Labs Blog](https://blog.secureflag.com/2022/11/04/threat-modelling-labs-now-available-on-secureflag/)
- [Infosec Institute Threat Model Lab](https://www.infosecinstitute.com/skills/courses/threat-model-lab/)