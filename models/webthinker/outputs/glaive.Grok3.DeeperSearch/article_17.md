

### Key Points
- It seems likely that the permissions inheritance issue is due to how macOS High Sierra handles SMB shares, particularly with Domain Admins group membership.
- Research suggests adjusting share permissions on the Windows Server 2019, such as removing "Full Control" for "Everyone" and explicitly granting "Full Control" to Domain Admins, may resolve the issue.
- The evidence leans toward checking and resetting NTFS permissions on the subfolder to ensure inheritance from the parent folder, especially for macOS clients.

### Direct Answer

To troubleshoot and resolve the permissions inheritance issue on your Windows Server 2019 file share, where Domain Admins are denied access to a subfolder from macOS High Sierra clients but not from Windows clients, follow these steps:

#### Adjust Share Permissions
First, check the share permissions on the Windows Server. It’s recommended to remove "Everyone" from the share permissions if present, as this can cause issues with macOS clients. Instead, add "Domain Admins" with "Full Control" to ensure administrators have complete access. You can also add "Authenticated Users" or "Domain Users" with "Change" or "Read" permissions, depending on your security needs.

#### Verify and Reset NTFS Permissions
Next, verify the NTFS permissions on the parent folder. Ensure "Domain Admins" have "Full Control" and that permission inheritance is enabled for subfolders and files. If the subfolder has broken inheritance, reset its permissions to inherit from the parent. This can be done by right-clicking the subfolder, selecting Properties, going to the Security tab, and choosing "Replace all child object permission entries with inheritable permission entries from this object."

#### Check for Specific Issues
There’s a possibility that macOS High Sierra adds a specific Security Identifier (SID, S-1-5-88-3-33216) to the ACLs, which might cause permission issues. Check the subfolder’s ACLs for this SID and consider removing it if it’s present, though this may require advanced permission management.

#### Test and Troubleshoot Further
After making these changes, test access from the macOS client to see if the issue is resolved. If it persists, use the "Effective Access" tool in Windows to check the actual permissions granted to the user on the subfolder. Also, review the Windows Server event logs for any permission-related errors when the macOS client attempts access.

An unexpected detail is that this issue is specific to macOS High Sierra and doesn’t affect Windows clients, suggesting a compatibility issue with how High Sierra handles SMB shares, which may require additional configuration or updates if possible.

---

### Survey Note: Detailed Analysis of Permissions Inheritance Issue on Windows Server 2019 with macOS High Sierra

This section provides a comprehensive analysis of the permissions inheritance issue on a Windows Server 2019 file share, where a subset of users with Domain Admins group membership are denied access to a specific subfolder from macOS High Sierra clients, while Windows clients remain unaffected. The investigation draws on various online resources and community discussions to offer a thorough understanding and actionable solutions.

#### Background and Problem Context
The issue involves a Windows Server 2019 file share where users, who are members of the Domain Admins group and have full control permissions on the parent folder, are denied access to a specific subfolder when accessing via macOS High Sierra (10.13) clients. This problem does not occur when the same users access the share from Windows clients, indicating a client-side compatibility issue with macOS High Sierra’s SMB implementation. Given the current date, March 26, 2025, and the age of High Sierra (released in 2017), it’s notable that the operating system may lack recent patches addressing SMB compatibility, especially with Windows Server 2019 (released in 2018).

#### Analysis of Permissions and Inheritance
Permissions in Windows Server are governed by both share permissions and NTFS permissions. Share permissions control access at the share level, while NTFS permissions, which include Access Control Lists (ACLs), manage file and folder access at the filesystem level. By default, subfolders and files inherit permissions from their parent folder unless inheritance is explicitly broken. However, the problem description suggests that inheritance is not functioning as expected for macOS High Sierra clients, potentially due to how these clients interact with the SMB protocol.

Research from community discussions, such as an Apple Community thread ([10.13.3 smb permissions inheritance lost](https://discussions.apple.com/thread/8337384)), highlights similar issues where copying or creating files on SMB shares from macOS High Sierra results in permissions being set such that only the creator can access them, ignoring inherited permissions. This thread suggests a workaround of removing "Full Control" for "Everyone" in share permissions, which implies adjusting share permissions to be more restrictive or specific to groups like Domain Admins.

Another relevant finding from a Server Fault question ([OSX 10.8.3 creating/editing files on Windows 7 share creates weird blocking account](https://serverfault.com/questions/498171/osx-10-8-3-creating-editing-files-on-windows-7-share-creates-weird-blocking-acco#498178)) indicates that macOS clients may add a specific SID (S-1-5-88-3-33216) to ACLs, related to Windows for UNIX translation, which can cause permission conflicts. This SID, storing UNIX mode information, might prevent access for certain users, including Domain Admins, when accessed from macOS.

#### Proposed Solutions and Steps
Based on the analysis, the following steps are recommended to troubleshoot and resolve the issue:

1. **Adjust Share Permissions:**
   - Access the share properties on the Windows Server via File Explorer or Computer Management.
   - Remove "Everyone" from the share permissions if present, as it may cause macOS clients to misinterpret permissions.
   - Add "Domain Admins" with "Full Control" to ensure administrators have complete access at the share level.
   - Optionally, add "Authenticated Users" or "Domain Users" with "Change" or "Read" permissions, depending on organizational security policies. This aligns with best practices to limit broad "Everyone" access, potentially improving compatibility with macOS.

2. **Verify and Reset NTFS Permissions:**
   - Check the NTFS permissions on the parent folder by right-clicking it, selecting Properties, and navigating to the Security tab. Ensure "Domain Admins" have "Full Control" and that the "Include inheritable permissions from this object's parent" option is enabled.
   - For the problematic subfolder, verify if inheritance is broken (indicated by a shield icon on the folder in advanced security settings). If broken, reset inheritance by selecting "Replace all child object permission entries with inheritable permission entries from this object" in the Security tab.

3. **Inspect and Manage Specific SIDs:**
   - Use the Advanced Security Settings for the subfolder to view the ACLs. Look for the SID S-1-5-88-3-33216, which may have been added by macOS High Sierra clients. If present, consider removing this entry, but note that this requires careful management to avoid affecting other users or functionalities. This step may involve using tools like `icacls` in Command Prompt for advanced ACL manipulation.

4. **Test Access and Monitor Logs:**
   - After making changes, test access from the macOS High Sierra client by attempting to access the subfolder. Verify if Domain Admins can now access it without denial.
   - Use the "Effective Access" tool in Windows (available in the Advanced Security Settings) to check the actual permissions granted to the user on the subfolder, ensuring no implicit denials.
   - Review the Windows Server event logs (Event Viewer under Windows Logs > Security) for any permission-related errors, such as access denied events (Event ID 4656 or 4670), when the macOS client attempts access.

5. **Consider Client-Side Adjustments:**
   - Ensure the macOS clients are updated to the latest version of High Sierra (10.13.6), as Apple may have released patches addressing SMB compatibility issues. Check for updates via System Preferences > Software Update.
   - Try accessing the share using the IP address instead of the hostname (e.g., `smb://192.168.x.x/share`) to rule out DNS-related issues, as suggested in community discussions ([Mac to Windows Share Issue](https://arstechnica.com/civis/threads/mac-to-windows-share-issue-and-general-insanity.1445785/)).

#### Additional Considerations
An unexpected detail is the client-specific behavior: the issue only manifests on macOS High Sierra, not Windows clients, suggesting a compatibility gap in how High Sierra’s SMB client (supporting SMB 3.0) interacts with Windows Server 2019’s SMB implementation (supporting SMB 3.1.1). This gap may stem from differences in handling ACLs, group memberships, or protocol features, as noted in various forums ([Marriott Library Mac Bug](https://apple.lib.utah.edu/mac-bug-connecting-to-non-apple-smb-shares/)).

Community discussions also mention workarounds like using CIFS (SMB1), but this is deprecated due to security vulnerabilities and not recommended, especially given Windows Server 2019’s default disabling of SMB1 ([Detect, enable, and disable SMBv1, SMBv2, and SMBv3](https://learn.microsoft.com/en-us/windows-server/storage/file-server/troubleshoot/detect-enable-and-disable-smbv1-v2-v3)). Another consideration is updating macOS to a newer version if possible, but given the problem’s context, this may not be feasible.

#### Table: Summary of Key Findings and Actions

| Aspect                  | Details                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| Affected OS             | macOS High Sierra (10.13), specifically when accessing Windows Server 2019 SMB shares |
| Server OS               | Windows Server 2019                                                    |
| Issue                   | Domain Admins denied access to subfolder, inheritance not working for macOS clients |
| Potential Cause         | macOS adding SID S-1-5-88-3-33216 to ACLs, breaking inheritance        |
| Recommended Actions     | Adjust share permissions, reset NTFS inheritance, check for specific SIDs, test access |
| Additional Checks       | Update macOS if possible, review event logs, use Effective Access tool |

#### Conclusion
The permissions inheritance issue likely arises from macOS High Sierra’s interaction with Windows Server 2019 SMB shares, potentially due to ACL handling and SID additions. By adjusting share and NTFS permissions as outlined, and ensuring proper inheritance, the issue should be resolvable. If problems persist, further investigation into event logs and client updates may be necessary, acknowledging the complexity of cross-platform compatibility in legacy systems.

### Key Citations
- [10.13.3 smb permissions inheritance lost](https://discussions.apple.com/thread/8337384)
- [OSX 10.8.3 creating/editing files on Windows 7 share creates weird blocking account](https://serverfault.com/questions/498171/osx-10-8-3-creating-editing-files-on-windows-7-share-creates-weird-blocking-acco#498178)
- [OSX - File Sharing - Permission problems](https://apple.stackexchange.com/questions/340690/osx-file-sharing-permission-problems)