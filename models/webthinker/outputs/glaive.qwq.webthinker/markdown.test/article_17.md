Question: How can I troubleshoot and resolve a permissions inheritance issue on a Windows Server 2019 file share, where a subset of users with the Domain Admins group membership are being denied access to a specific subfolder despite having full control permissions on the parent folder, and the issue only occurs when the users are accessing the share via a macOS client running High Sierra, while Windows clients are unaffected?

# Troubleshooting Domain Admins Access Denial on Windows Server 2019 File Share via macOS High Sierra Clients

## Introduction

The problem at hand involves Domain Admins being denied access to a specific subfolder on a Windows Server 2019 file share when accessing it via macOS High Sierra clients, despite having Full Control permissions on the parent folder. This issue is particularly perplexing because Domain Admins typically possess elevated privileges, which should grant them unrestricted access to the file share and its subfolders. The discrepancy arises due to a combination of factors, including interactions between NTFS permissions inheritance, SMB protocol compatibility between Windows Server 2019 and macOS High Sierra, and potential misconfigurations on either the server or client side.

### Understanding the Problem

1. **NTFS Permissions Inheritance**:
   - **Inheritance Breakage**: One common cause of access issues is the breakage of NTFS permissions inheritance. If the subfolder has inheritance disabled, it will not inherit permissions from the parent folder, even if the parent folder grants Full Control to Domain Admins. This can lead to situations where the subfolder retains its own explicit permissions, which may not include the necessary access for Domain Admins.
   - **Explicit Deny Entries**: Another potential issue is the presence of explicit "Deny" entries on the subfolder. Even if the parent folder grants Full Control, an explicit "Deny" entry on the subfolder will override all other permissions, effectively blocking access for Domain Admins.

2. **SMB Protocol Compatibility**:
   - **SMB Version Support**: macOS High Sierra supports SMBv2 and SMBv3, but it may not fully support all features of SMBv3, such as encryption or signing. Windows Server 2019, on the other hand, defaults to SMBv3 and may enforce stricter security settings, such as requiring encryption or mandatory signing. This mismatch can lead to connection failures or access denials when macOS clients attempt to connect to the file share.
   - **Authentication Methods**: macOS High Sierra primarily uses NTLM authentication for SMB, while Windows Server 2019 relies heavily on Kerberos for domain authentication. If the server is configured to require Kerberos, macOS clients may face authentication issues, especially if the server’s DNS name is not used for connections.

3. **Client and Server Configurations**:
   - **Server-Side Settings**: The server’s configuration, including Group Policy Objects (GPOs), can restrict Domain Admins’ access. For example, GPOs that remove Domain Admins from local groups or apply conflicting permissions can block access to the subfolder. Additionally, the server’s SMB settings, such as requiring encryption or signing, can prevent macOS clients from connecting.
   - **Client-Side Settings**: macOS High Sierra clients may have configuration settings that affect SMB connectivity. For instance, the `/etc/nsmb.conf` file can be used to disable SMB signing or specify preferred SMB versions. Misconfigurations in this file can lead to authentication failures or protocol negotiation issues.

### Scope and Objectives

This article aims to provide a comprehensive guide for troubleshooting and resolving the access issues faced by Domain Admins when using macOS High Sierra clients to access a Windows Server 2019 file share. The objectives include:

1. **Identifying the Root Cause**: By examining NTFS permissions, SMB protocol settings, and client/server configurations, we will identify the specific factors contributing to the access denials.
2. **Step-by-Step Troubleshooting**: We will outline a series of steps to diagnose and resolve the issue, including checking permissions, verifying SMB settings, and adjusting client configurations.
3. **Best Practices**: We will provide best practices for configuring both the server and client to ensure smooth and secure access to the file share.

### Structure of the Article

The article is structured as follows:

1. **Introduction**: This section provides an overview of the problem, its potential causes, and the objectives of the article.
2. **Background on NTFS Permissions and Inheritance**: A detailed explanation of NTFS permissions, inheritance, and common issues that can lead to access denials.
3. **Background on SMB Protocol and macOS Compatibility**: An exploration of SMB protocol versions, security features, and compatibility issues between Windows Server 2019 and macOS High Sierra.
4. **Troubleshooting Steps**: A step-by-step guide for diagnosing and resolving access issues, including server-side and client-side configurations.
5. **Best Practices**: Recommendations for configuring the server and client to prevent future access issues.
6. **Conclusion**: A summary of the key points and a final note on maintaining secure and reliable file share access.

## Background on NTFS Permissions and Inheritance

### NTFS Permissions Overview

NTFS (New Technology File System) permissions are a critical component of Windows file and folder security. They determine who can access, modify, or delete files and folders on a Windows system. NTFS permissions are hierarchical, meaning that subfolders and files can inherit permissions from their parent folders. This inheritance model simplifies permission management by allowing administrators to set permissions at a higher level and have them propagate to lower levels.

### Inheritance Mechanism

Inheritance in NTFS permissions works as follows:
- **Inherited Permissions**: When a folder is created, it inherits the permissions of its parent folder. This means that any user or group with access to the parent folder will also have the same access to the subfolder, unless explicitly modified.
- **Explicit Permissions**: Administrators can set explicit permissions on a subfolder or file, which override inherited permissions. This is useful for fine-grained control over specific resources.
- **Disabling Inheritance**: Inheritance can be disabled on a subfolder, which stops it from inheriting permissions from its parent. When inheritance is disabled, the subfolder retains its current permissions, and any changes to the parent folder's permissions do not affect it.

### Domain Admins and Elevated Privileges

Domain Admins are a special group in Active Directory that have extensive administrative privileges across the domain. By default, Domain Admins are members of the local Administrators group on all domain-joined machines. This membership grants them Full Control permissions, allowing them to manage files, folders, and system settings with minimal restrictions. However, if a subfolder has inheritance disabled or contains explicit Deny permissions, even Domain Admins can be restricted from accessing it.

### Common Scenarios Leading to Access Denials

1. **Inheritance Disabled**:
   - If a subfolder has inheritance disabled, it will not receive updates to the parent folder's permissions. This can lead to a situation where Domain Admins, who rely on inherited permissions, lose access to the subfolder.
   - **Solution**: Re-enable inheritance on the subfolder to ensure it receives the necessary permissions from the parent.

2. **Explicit Deny Permissions**:
   - An explicit Deny permission on a subfolder or file will override all other permissions, including those granted to Domain Admins. This can be a common cause of access denials, even for users with Full Control.
   - **Solution**: Remove or modify the explicit Deny permissions to allow Domain Admins access.

3. **Ownership Issues**:
   - If a subfolder is owned by a different user or group, and that owner has not granted appropriate permissions, Domain Admins may be unable to access it.
   - **Solution**: Take ownership of the subfolder and reset permissions to ensure Domain Admins have the necessary access.

## Background on SMB Protocol and macOS Compatibility

### SMB Protocol Overview

SMB (Server Message Block) is the primary protocol used for file sharing in Windows environments. Windows Server 2019 supports SMBv3, which includes enhanced security features such as encryption and signing. These features are designed to improve data integrity and security but can sometimes cause compatibility issues with older clients.

### SMBv3 and macOS High Sierra

macOS High Sierra supports SMBv2 and SMBv3.0 but lacks full compatibility with SMBv3.1.1, which is used by Windows Server 2019. This version disparity can lead to the following issues:
- **Encryption**: If the server enforces SMBv3.1.1 encryption, macOS High Sierra clients may be unable to connect, as they do not fully support this feature.
- **Signing**: macOS High Sierra defaults to requiring packet signing, which can conflict with the server's configuration. Disabling signing on the client or server can resolve this issue.
- **Authentication**: macOS may use NTLM authentication by default, which can fail if the server requires Kerberos. Using the server’s DNS name instead of an IP address can help resolve authentication issues.

### Troubleshooting Steps

1. **Check Inheritance Status**:
   - Verify if the subfolder has inheritance disabled. If so, re-enable it to ensure it receives the necessary permissions from the parent folder.
   - **Steps**: Right-click the subfolder > Properties > Security tab > Advanced > Enable Inheritance.

2. **Review Explicit Permissions**:
   - Check for explicit Deny permissions on the subfolder. Remove or modify these permissions to allow Domain Admins access.
   - **Steps**: Right-click the subfolder > Properties > Security tab > Advanced > Permissions.

3. **Adjust SMB Settings**:
   - Ensure the server allows SMBv3.0 and does not enforce SMBv3.1.1 encryption. Adjust the server’s SMB settings to accommodate macOS High Sierra clients.
   - **Steps**: Use PowerShell cmdlets like `Set-SmbServerConfiguration` to modify protocol settings.

4. **Client Configuration**:
   - Modify the macOS client’s `/etc/nsmb.conf` file to disable signing and specify preferred SMB versions.
   - **Steps**: Add the following settings to `/etc/nsmb.conf`:
     ```ini
     [default]
     signing_required=no
     protocol_vers_map=4
     ```

## Possible Causes of Access Denial

The access denial for Domain Admins via macOS High Sierra clients can stem from several factors. Each of these factors must be evaluated systematically to pinpoint the root cause. Below is a detailed exploration of the potential causes:

### 1. Permissions Inheritance Breakage

**Inheritance Mechanism**:
- **Inherited Permissions**: By default, subfolders inherit permissions from their parent folders. This hierarchical structure ensures that permissions set on a parent folder are applied to all subfolders and files within it.
- **Disabling Inheritance**: If inheritance is disabled on a subfolder, it stops inheriting permissions from the parent. This can lead to access issues if the subfolder’s explicit permissions do not include the necessary rights for Domain Admins.

**Explicit Deny Entries**:
- **Deny Permissions**: An explicit "Deny" entry on the subfolder, either inherited or set directly, will override all other permissions, including those granted to Domain Admins. This can result in access denials even if the parent folder grants Full Control.
- **Checking Inheritance**: To verify if inheritance is disabled, right-click the subfolder, go to **Properties > Security > Advanced**, and check the **Enable Inheritance** section. If inheritance is disabled, re-enable it to apply parent permissions.

### 2. Share Permissions Restriction

**Share Permissions vs. NTFS Permissions**:
- **Share Permissions**: These control network access to the shared folder. They are separate from NTFS permissions and must be configured to grant access to the desired users or groups.
- **Network Access**: If Domain Admins are not explicitly included in the share permissions, they may be blocked from accessing the subfolder, even if NTFS permissions grant Full Control.

**Verifying Share Permissions**:
- **Advanced Sharing**: Open the subfolder’s **Properties > Sharing > Advanced Sharing**. Ensure that the share permissions include Domain Admins with "Full Control."
- **Inherited Share Permissions**: If share permissions are inherited from the parent folder, confirm that the parent’s share permissions grant sufficient access.

### 3. SMBv3 Encryption Requirements

**SMBv3 Features**:
- **Encryption**: Windows Server 2019 may enforce SMBv3 encryption, which enhances security but can cause compatibility issues with older clients like macOS High Sierra.
- **macOS Compatibility**: macOS High Sierra supports SMBv2 and SMBv3.0 but lacks full compatibility with SMBv3.1.1, which includes encryption enhancements.

**Disabling Encryption**:
- **Server Configuration**: On the Windows Server, ensure that SMBv3 encryption is not required. This can be done via Group Policy or the registry.
- **Registry Key**: Modify the registry key `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\LanmanServer\Parameters` and set `EncryptData` to `0` (DWORD) to disable encryption.

### 4. Packet Signing Mismatch

**Packet Signing**:
- **Mandatory Signing**: The server might require mandatory signing for SMB connections, which can cause issues if the client does not support it.
- **macOS Configuration**: macOS High Sierra defaults to requiring packet signing. This can be disabled to improve compatibility with the server.

**Disabling Signing on macOS**:
- **nsmb.conf**: Edit the `/etc/nsmb.conf` file to set `signing_required=no`. This can be done via Terminal:
  ```bash
  sudo nano /etc/nsmb.conf
  ```
  Add the following lines:
  ```ini
  [default]
  signing_required=no
  ```
- **Save and Apply**: Save the file and restart the Mac or log out and back in to apply the changes.

### 5. Authentication Method Conflict

**Authentication Methods**:
- **NTLM vs. Kerberos**: macOS High Sierra may default to NTLM authentication, which can conflict with the server’s Kerberos requirements.
- **Using DNS Name**: Using the server’s DNS name (instead of IP) can force Kerberos authentication, which is more secure and compatible with modern SMB versions.

**Testing Authentication**:
- **Connect via DNS**: When connecting to the server, use the server’s DNS name (e.g., `smb://server.example.com/share`) instead of the IP address.
- **Manual Credentials**: Test with explicit credentials via the command line:
  ```bash
  mount_smbfs //domain\\admin@server.example.com/share /local/mount/path
  ```

### 6. Group Policy Restrictions

**Group Policy Objects (GPOs)**:
- **Restrictive Policies**: GPOs applied to the server’s OU can restrict Domain Admins’ logon rights or limit access to specific protocols.
- **Common Restrictions**: Policies like **Deny log on through Remote Interface** or **Access this computer from the network** can block Domain Admins unless explicitly allowed.

**Checking GPOs**:
- **Group Policy Management Console (GPMC)**: Use GPMC to review GPOs applied to the server’s OU. Ensure that no policies are restricting Domain Admins’ access.
- **Effective Access**: Use the **Effective Access** tool to audit permissions for the Domain Admin account and identify conflicting settings.

### 7. Invalid SIDs or Ownership Issues

**SID and Ownership**:
- **Invalid SIDs**: The subfolder’s ownership might be assigned to an invalid SID (e.g., from a prior domain migration), preventing Domain Admins from gaining access.
- **Taking Ownership**: If the subfolder’s ownership is incorrect, take ownership of the folder and subfolders, then reset permissions.

**Steps to Take Ownership**:
- **Advanced Security Settings**: Right-click the subfolder, go to **Properties > Security > Advanced > Owner**. Click **Edit** and select the appropriate account (e.g., `Administrators`).
- **Reset Permissions**: After taking ownership, reset permissions to ensure Domain Admins have the necessary rights.

### 8. Access-Based Enumeration (ABE)

**ABE Settings**:
- **Hiding Subfolders**: ABE settings on the share might hide the subfolder from users lacking permissions, including Domain Admins if misconfigured.
- **Disabling ABE**: Temporarily disable ABE to test whether the issue is related to enumeration or actual permissions.

**Disabling ABE**:
- **Share Properties**: Open the share’s properties and go to **Security > Advanced > Auditing**. Disable ABE if it is enabled and test access.

### 9. Firewall/Network Policies Blocking Traffic

**Firewall Rules**:
- **Blocking Traffic**: macOS clients might be blocked by firewall rules on the server or network devices, particularly for newer SMB ports or protocols.
- **Common Ports**: Ensure that TCP port 445 (SMB) and UDP ports 137/138 (NetBIOS) are open between the macOS client and the Windows Server.

**Checking Firewall Settings**:
- **Server Firewall**: Use the Windows Firewall with Advanced Security to verify that the necessary ports are open.
- **Network Devices**: Check firewall rules on network devices (e.g., routers, switches) to ensure they are not blocking SMB traffic.

By systematically evaluating these factors, you can identify and resolve the root cause of the access denial issue for Domain Admins using macOS High Sierra clients. Each step should be carefully documented to ensure a thorough and effective troubleshooting process.

## Troubleshooting Steps for Domain Admins Access Issues

To diagnose and resolve the access denial for Domain Admins when accessing a specific subfolder on a Windows Server 2019 file share via macOS High Sierra clients, follow these structured steps:

### **Step 1: Verify Subfolder Permissions**

#### **Check NTFS Permissions**
1. **Navigate to the Subfolder**:
   - Right-click the subfolder in File Explorer.
   - Select **Properties**.
   - Go to the **Security** tab.

2. **Review Permissions**:
   - Ensure that **Domain Admins** are listed with **Full Control** permissions.
   - If not, click **Edit** to add or modify the permissions.
   - Click **Add** to include **Domain Admins** and assign **Full Control**.

3. **Check Inheritance**:
   - Click **Advanced** to open the **Advanced Security Settings**.
   - Check the **Inheritance** section to ensure that the subfolder is inheriting permissions from the parent folder.
   - If inheritance is disabled, click **Change permissions**.
   - Check the box **Include inheritable permissions from this object's parent**.
   - Click **Apply** and **OK** to propagate changes.

4. **Remove Deny Entries**:
   - In the **Advanced Security Settings**, review the **Permissions** entries.
   - Look for any **Deny** entries affecting **Domain Admins**.
   - Remove or modify these entries to ensure they do not override inherited permissions.

#### **Check Share Permissions**
1. **Navigate to the Share**:
   - Open **Server Manager**.
   - Go to **File and Storage Services** > **Shares**.
   - Right-click the share and select **Properties**.
   - Go to the **Share** tab and click **Advanced Sharing**.

2. **Review Share Permissions**:
   - Click **Permissions**.
   - Ensure that **Domain Admins** are listed with **Full Control** share permissions.
   - If not, click **Add** to include **Domain Admins** and assign **Full Control**.
   - Click **Apply** and **OK** to save changes.

### **Step 2: Inspect Windows Server Event Logs**

#### **Event Viewer Filters**
1. **Open Event Viewer**:
   - Press **Windows + R** and type `eventvwr.msc`.
   - Press **Enter** to open Event Viewer.

2. **Navigate to Security Logs**:
   - Go to **Windows Logs** > **Security**.

3. **Filter for Relevant Events**:
   - Click **Filter Current Log** in the Actions pane.
   - In the **Filter Current Log** dialog, add the following Event IDs:
     - **4663 (File System Access)**
     - **5145 (File Share Access)**
     - **4625 (Logon Failure)**
   - Click **OK** to apply the filter.

4. **Review Events**:
   - Look for failed access attempts originating from the macOS client’s IP address.
   - Check the **Details** tab for each event to identify the **Account Name**, **Object Name**, and **Access Mask**.
   - Event 4625 may indicate authentication issues, such as incorrect credentials or protocol mismatches.

### **Step 3: Audit macOS Client-Side Configuration**

#### **Check SMB Version**
1. **Verify Negotiated SMB Version**:
   - Open **Terminal** on the macOS client.
   - Run the following command to check the negotiated SMB version:
     ```bash
     smbutil proto //server/share
     ```

2. **Configure `/etc/nsmb.conf`**:
   - Open the `/etc/nsmb.conf` file in a text editor:
     ```bash
     sudo nano /etc/nsmb.conf
     ```
   - Add the following settings to enforce SMBv3 and disable signing if the server allows:
     ```ini
     [default]
     protocol_vers_map=4  # Enforces SMBv3
     signing_required=no   # Disable signing if server allows
     ```
   - Save the file and exit the editor.

#### **Test Explicit Credentials**
1. **Connect Using Command Line**:
   - Open **Terminal** and use the following command to connect to the subfolder with explicit credentials:
     ```bash
     open smb://Domain\\AdminUsername@ServerIP/share/subfolder
     ```
   - This bypasses Keychain cached credentials and ensures the correct domain and username are used.

### **Step 4: Test SMB Protocol Compatibility**

#### **On the Server**
1. **Ensure SMBv3 is Enabled**:
   - Open **Server Manager**.
   - Go to **File and Storage Services** > **File Shares**.
   - Click **SMB Shares** and ensure **SMB 3.0** is enabled.

2. **Disable SMB Signing**:
   - Open **Group Policy Management**.
   - Navigate to **Computer Configuration** > **Policies** > **Administrative Templates** > **Network** > **Lanman Workstation**.
   - Double-click **Enable insecure guest logons** and set it to **Enabled**.
   - Navigate to **Computer Configuration** > **Policies** > **Administrative Templates** > **Network** > **Lanman Server**.
   - Double-click **Configure signing and encryption** and set it to **Disabled**.
   - Apply the policy and restart the server if necessary.

3. **Allow SMBv2**:
   - If SMBv3 is not compatible, temporarily allow **SMBv2** on the server:
     - Open **PowerShell** as an administrator.
     - Run the following command:
       ```powershell
       Set-SmbServerConfiguration -EnableSMB2Protocol $true
       ```

#### **On macOS**
1. **Disable SMB Signing**:
   - Ensure the `/etc/nsmb.conf` file is configured as follows:
     ```ini
     [default]
     protocol_vers_map=4  # Enforces SMBv3
     signing_required=no   # Disable signing if server allows
     ```

2. **Update macOS**:
   - Ensure the macOS client is updated to the latest High Sierra patch (10.13.6) for better SMB compatibility.

### **Step 5: Review Group Policy Settings**

#### **Restricted Groups**
1. **Check GPOs**:
   - Open **Group Policy Management**.
   - Navigate to the Organizational Unit (OU) where the server is located.
   - Review GPOs to ensure **Domain Admins** are not removed from the local **Administrators** group.

#### **User Rights Assignment**
1. **Confirm User Rights**:
   - Open **Local Security Policy** (or **Group Policy Management**).
   - Navigate to **Local Policies** > **User Rights Assignment**.
   - Ensure **Domain Admins** have the following rights:
     - **Take ownership of files or other objects**
     - **Back up files and directories**

### **Step 6: Reset Subfolder Permissions**

#### **Force Inheritance**
1. **Use `icacls`**:
   - Open **PowerShell** as an administrator.
   - Run the following command to reset permissions recursively:
     ```powershell
     icacls "C:\Path\To\Subfolder" /reset /t
     ```

2. **Re-enable Inheritance**:
   - Right-click the subfolder in File Explorer.
   - Select **Properties**.
   - Go to the **Security** tab and click **Advanced**.
   - Click **Disable inheritance**.
   - Choose **Convert inherited permissions to explicit**.
   - Reapply **Full Control** to **Domain Admins**.
   - Click **Apply** and **OK** to save changes.

### **Step 7: Test Local Access on Windows Server**

1. **Direct Access**:
   - Log in to the server via a Windows client.
   - Attempt to access the subfolder directly.
   - If access is successful, the issue is likely macOS-specific.

### **Step 8: Validate Network Connectivity**

#### **Port Check**
1. **Test Network Connection**:
   - Open **PowerShell** on the server.
   - Run the following command to test connectivity:
     ```powershell
     Test-NetConnection -ComputerName ServerIP -Port 445
     ```
   - Ensure **TCP 445** and **UDP 137/138** are open between the macOS client and the server.

#### **Firewall Rules**
1. **Temporarily Disable Firewalls**:
   - Temporarily disable firewalls on both the server and the macOS client to test if they block SMB traffic.
   - If the issue is resolved, re-enable the firewalls and configure rules to allow SMB traffic.

### **Step 9: Check Ownership and SIDs**

1. **Take Ownership**:
   - Right-click the subfolder in File Explorer.
   - Select **Properties**.
   - Go to the **Security** tab and click **Advanced**.
   - Click **Change** next to **Owner**.
   - Enter the name of a valid domain account (e.g., **SERVER\Administrators**).
   - Check the box **Replace owner on subcontainers and objects**.
   - Click **Apply** and **OK** to save changes.

2. **Reset Permissions**:
   - Ensure the subfolder’s permissions are reset to the parent’s ACLs.
   - Use **icacls** to reset permissions recursively:
     ```powershell
     icacls "C:\Path\To\Subfolder" /reset /t
     ```

### **Step 10: Evaluate Authentication Methods**

1. **Use DNS Name**:
   - Ensure the macOS client uses the server’s **DNS name** (not IP) to force Kerberos authentication.
   - Connect using the following command:
     ```bash
     open smb://server.domain.com/share/subfolder
     ```

2. **Disable Kerberos**:
   - Temporarily disable **Kerberos** on the server to test if **NTLM** resolves the issue:
     - Open **Group Policy Management**.
     - Navigate to **Computer Configuration** > **Policies** > **Administrative Templates** > **Network** > **Lanman Workstation**.
     - Double-click **Enable insecure guest logons** and set it to **Enabled**.
     - Apply the policy and restart the server if necessary.

By methodically executing these steps, administrators can isolate the cause of the access denial and implement targeted fixes to resolve the issue.

## Resolution Strategies

Based on the identified causes, the following resolutions can be implemented to resolve the access denial issue for Domain Admins when accessing a specific subfolder on a Windows Server 2019 file share via macOS High Sierra clients.

### **A. Fix Permissions Inheritance**

#### **Re-enable Inheritance**
1. **Navigate to the Subfolder’s Advanced Security Settings**:
   - Right-click the subfolder > **Properties** > **Security** tab > **Advanced**.
   - Click **Change permissions**.
   - Check the box **Include inheritable permissions from this object's parent**.
   - Click **Apply** and then **OK** to propagate the changes.
   - If prompted, choose **Replace all child object permissions with inheritable permissions from this object** to ensure all subfolders and files inherit the parent’s permissions.

#### **Manually Apply Permissions**
1. **Explicitly Grant Full Control**:
   - If inheritance is impractical or if you need to set specific permissions, manually grant **Full Control** to Domain Admins on the subfolder.
   - Right-click the subfolder > **Properties** > **Security** tab > **Edit**.
   - Click **Add** and enter the Domain Admins group.
   - Check the **Full Control** permission and click **Apply** and **OK**.

### **B. Adjust SMB Protocol Settings**

#### **On Windows Server 2019**
1. **Disable Require SMB Encryption**:
   - Open **Group Policy Management Console (GPMC)**.
   - Navigate to **Computer Configuration > Administrative Templates > Network > Lanman Workstation**.
   - Find the policy **Require encryption for data transfers** and set it to **Disabled**.
   - Apply the policy and restart the server if necessary.

2. **Allow SMBv2**:
   - If High Sierra clients cannot negotiate SMBv3, allow SMBv2 on the server.
   - Open **Server Manager** > **File and Storage Services** > **File Shares**.
   - Right-click the share and select **Properties**.
   - Go to the **Share** tab and click **Advanced Sharing**.
   - Click **Permissions** and ensure **Domain Admins** have **Full Control**.
   - In the **SMB Settings** section, ensure **SMB 2.x** is enabled.

#### **On macOS High Sierra**
1. **Configure `/etc/nsmb.conf`**:
   - Open Terminal and edit or create the `/etc/nsmb.conf` file:
     ```bash
     sudo nano /etc/nsmb.conf
     ```
   - Add the following lines to disable signing and prioritize SMBv3:
     ```ini
     [default]
     signing_required=no
     protocol_vers_map=4
     ```
   - Save the file and exit the editor.
   - Restart the SMB service or the macOS client to apply changes:
     ```bash
     sudo killall -HUP mDNSResponder
     ```

### **C. Correct Share Permissions**

1. **Explicitly Grant Full Control**:
   - Navigate to the parent share’s **Properties** > **Sharing** tab > **Advanced Sharing**.
   - Click **Permissions** and ensure **Domain Admins** have **Full Control**.
   - Click **Apply** and **OK** to save the changes.
   - Verify that the subfolder inherits these permissions by checking its **Security** tab.

### **D. Update macOS Client Software**

1. **Install the Latest macOS High Sierra Updates**:
   - Open **System Preferences** > **Software Update**.
   - Install any available updates to ensure your macOS High Sierra is up to date (e.g., **10.13.6**).
   - This can improve SMBv3 compatibility and resolve known issues.

### **E. Audit and Modify Group Policies**

1. **Review GPOs for Conflicting Policies**:
   - Open **Group Policy Management Console (GPMC)**.
   - Navigate to the GPOs applied to the server’s OU.
   - Check for policies that restrict Domain Admins’ logon rights (e.g., **Deny log on through Remote interface**).
   - Adjust or remove conflicting policies to ensure Domain Admins have the necessary access.

### **F. Bypass Keychain Caching**

1. **Remove Cached Credentials**:
   - Open **Keychain Access** (System Preferences > Keychain Access).
   - Select the **Login** keychain and search for the server’s credentials.
   - Delete any cached credentials for the server.
   - Reconnect to the server using explicit credentials to ensure fresh authentication.

### **G. Use Alternative Authentication Methods**

1. **Configure Server to Allow NTLM**:
   - If Kerberos fails, configure the server to allow NTLM authentication temporarily.
   - Open **Group Policy Management Console (GPMC)**.
   - Navigate to **Computer Configuration > Administrative Templates > Network > Lanman Workstation**.
   - Find the policy **Send unencrypted password to third-party SMB servers** and set it to **Enabled**.
   - Apply the policy and restart the server if necessary.

2. **Use `mount_smbfs` Command**:
   - For macOS clients, use the `mount_smbfs` command with explicit credentials:
     ```bash
     sudo mount_smbfs //<Domain%5CAdminUsername>:<Password>@ServerIP/share/subfolder /Volumes/LocalMountPoint
     ```

### **H. Disable Access-Based Enumeration (ABE)**

1. **Uncheck ABE in Share Properties**:
   - Navigate to the share’s **Properties** > **Sharing** tab > **Advanced Sharing**.
   - Click **Permissions** and ensure **Domain Admins** have **Full Control**.
   - Uncheck the box **Enable Access-Based Enumeration (ABE)** to ensure the subfolder is visible to Domain Admins.
   - Click **Apply** and **OK** to save the changes.

### **I. Check and Fix Ownership Issues**

1. **Take Ownership of the Subfolder**:
   - Right-click the subfolder > **Properties** > **Security** tab > **Advanced**.
   - Click **Change** next to the **Owner** field.
   - Enter the Domain Admins group and click **OK**.
   - Check the box **Replace owner on subcontainers and objects**.
   - Click **Apply** and **OK** to save the changes.
   - Grant **Full Control** to Domain Admins if necessary.

### **J. Test and Document Changes**

1. **Verify Access**:
   - After applying the fixes, verify access via both macOS High Sierra and Windows clients.
   - Ensure that Domain Admins can access the subfolder without issues.

2. **Document Successful Configurations**:
   - Document the changes made and the steps taken to resolve the issue.
   - This documentation can be useful for future reference and to prevent recurrence of the problem.

These strategies address both permissions and protocol-related causes, ensuring cross-platform access for Domain Admins. By methodically implementing these solutions, administrators can resolve the access denial issue and maintain a secure and functional file share environment.

## Conclusion

The access denial experienced by Domain Admins on macOS High Sierra clients is a multifaceted issue that arises from a combination of NTFS permissions inheritance breaks and SMB protocol compatibility gaps between Windows Server 2019 and older macOS versions. Addressing this problem requires a systematic approach that encompasses both permissions management and protocol configuration adjustments.

### Permissions Inheritance and Explicit Denies

One of the primary causes of access denial is the breakage of permissions inheritance on the subfolder. When inheritance is disabled, the subfolder does not inherit the Full Control permissions granted to Domain Admins on the parent folder. This can be resolved by re-enabling inheritance and propagating the parent folder's permissions to the subfolder. Additionally, explicit Deny permissions on the subfolder can override inherited Allow permissions, leading to access denials. Administrators should carefully review and remove any Deny entries affecting Domain Admins to ensure they have the necessary access.

### Share Permissions and Network Access

Share permissions, which control network-level access, can also restrict Domain Admins even if NTFS permissions are correctly configured. Ensuring that Domain Admins are explicitly included in the share permissions with Full Control is crucial. This can be verified and adjusted through the **Advanced Sharing** dialog in the folder's properties. Misconfigurations in share permissions can often be overlooked, but they play a significant role in network access control.

### SMB Protocol Compatibility

SMB protocol compatibility is another critical factor. Windows Server 2019 enforces stricter security features in SMBv3, such as encryption and mandatory signing, which may not be fully supported by macOS High Sierra. Disabling these requirements on the server can help resolve compatibility issues. For example, disabling SMB encryption and signing requirements via Group Policy can allow macOS clients to connect without encountering protocol mismatches. Additionally, configuring macOS clients to use SMBv3.0 (not SMBv3.1.1) by editing the `/etc/nsmb.conf` file ensures that the client and server can negotiate a compatible protocol version.

### Authentication Methods

Authentication methods can also contribute to access issues. macOS High Sierra may default to NTLM authentication, which can conflict with the server's Kerberos requirements. Using the server's DNS name instead of its IP address can force Kerberos authentication, which is more secure and often required by modern Windows servers. If Kerberos fails, temporarily allowing NTLM authentication on the server can help diagnose and resolve the issue. Testing explicit credentials via the command line can bypass Keychain caching and ensure that the correct credentials are being used.

### Group Policy and Ownership

Group Policy Objects (GPOs) can restrict Domain Admins' access if they are removed from the local Administrators group or if specific user rights are denied. Reviewing and modifying GPOs to ensure that Domain Admins have the necessary rights, such as taking ownership of files and backing up data, is essential. Ownership issues can also prevent Domain Admins from modifying permissions. Taking ownership of the subfolder and resetting permissions to the parent's ACLs can resolve these issues.

### Regular Audits and Documentation

Regular audits of permissions, GPOs, and event logs are crucial for maintaining stable cross-platform access. Event logs, particularly Event IDs 4663 and 5145, provide valuable insights into access attempts and failures. Auditing permissions and GPOs can help identify and correct misconfigurations that may lead to access denials. Documenting successful configurations and changes ensures that administrators can quickly resolve similar issues in the future.
## Background on NTFS Permissions and Inheritance

### NTFS Permissions Overview
NTFS (New Technology File System) permissions on Windows Server 2019 are a critical component of file and folder access control. These permissions define who can perform specific actions on files and directories, such as reading, writing, executing, or modifying. NTFS permissions are hierarchical, meaning that subfolders and files within a parent folder typically inherit the permissions of their parent unless explicitly modified. This inheritance mechanism ensures that permissions are applied consistently across the file system, simplifying management and reducing the risk of security vulnerabilities.

### Inheritance Mechanism
Inheritance in NTFS permissions works by propagating the permissions of a parent folder to its subfolders and files. When a new subfolder or file is created within a parent folder, it automatically inherits the permissions of the parent. This ensures that the new item has the same access control as its parent, maintaining a consistent security policy. However, inheritance can be disabled on a subfolder or file, allowing for the application of unique permissions that do not depend on the parent. Disabling inheritance can be useful in scenarios where specific access controls are required for a particular subfolder or file, but it can also lead to access issues if not managed carefully.

### Domain Admins and Elevated Privileges
Domain Admins are a special group in Active Directory that have extensive administrative rights across the domain. By default, Domain Admins are members of the **local Administrators group** on all domain-joined machines, including Windows Server 2019. This membership grants them **Full Control** permissions, allowing them to perform almost any action on the server, including managing files and folders, configuring system settings, and managing user accounts. However, if a subfolder has inheritance disabled or explicit permissions applied, these elevated rights may not propagate, leading to access denials even for Domain Admins.

### Common Scenarios Leading to Access Denials
Access denials for Domain Admins can occur in several scenarios:
- **Inheritance Disabled**: If a subfolder has inheritance disabled, it will not inherit the permissions of its parent. This can result in Domain Admins lacking the necessary permissions to access the subfolder, even if they have Full Control on the parent.
- **Explicit Deny Permissions**: If a subfolder has explicit Deny permissions set, these will override any inherited Allow permissions, including those granted to Domain Admins.
- **Share Permissions**: Share permissions, which control network access to a folder, can also restrict access. If Domain Admins are not explicitly included in the share permissions, they may be blocked from accessing the subfolder, even if NTFS permissions grant them Full Control.
- **Group Policy Restrictions**: Group Policy Objects (GPOs) can enforce restrictions on user and group access. If a GPO restricts Domain Admins' logon rights or access to specific protocols, it can prevent them from accessing the subfolder.

### SMB Protocol and macOS Compatibility
The **Server Message Block (SMB)** protocol is the primary mechanism for file sharing between Windows and macOS systems. Windows Server 2019 defaults to **SMBv3**, which includes advanced security features such as encryption and mandatory signing. These features enhance security but can also introduce compatibility issues with older clients, such as macOS High Sierra.

### SMBv3 and macOS High Sierra
macOS High Sierra supports **SMBv2** and **SMBv3.0**, but lacks full compatibility with **SMBv3.1.1**, which was introduced in later versions of Windows. This version disparity can lead to several issues:
- **Encryption**: If the server enforces **SMBv3 encryption**, macOS High Sierra clients may be unable to connect, as they do not support the required encryption standards.
- **Packet Signing**: The server may require **mandatory signing** for SMB connections, which macOS clients may not negotiate properly, leading to authentication failures.
- **Authentication Methods**: macOS High Sierra may default to **NTLM authentication**, which can conflict with the server’s **Kerberos** requirements. Using the server’s DNS name instead of its IP address can force Kerberos authentication, resolving this issue.

### The Intersection of NTFS and SMB
The issue of Domain Admins being denied access to a subfolder on a Windows Server 2019 file share via macOS High Sierra clients arises at the intersection of these two systems:
- **Permissions Inheritance**: If the subfolder has inheritance disabled or explicit Deny permissions, Domain Admins may be blocked despite having Full Control on the parent folder.
- **SMB Protocol Compatibility**: macOS High Sierra’s limitations with SMBv3 features, such as encryption and signing, can cause authentication failures or permission denials when interacting with the server.

By understanding these underlying mechanisms, administrators can better diagnose and resolve access issues, ensuring that Domain Admins have the necessary permissions to manage files and folders across the network.

