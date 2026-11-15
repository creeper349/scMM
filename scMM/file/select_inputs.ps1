Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace ScmmPicker {
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct Filter {
        [MarshalAs(UnmanagedType.LPWStr)] public string Name;
        [MarshalAs(UnmanagedType.LPWStr)] public string Pattern;
    }
    [ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface Item {
        void BindToHandler(IntPtr ctx, ref Guid handler, ref Guid iid, out IntPtr result);
        void GetParent(out Item parent);
        void GetDisplayName(uint kind, out IntPtr name);
    }
    [ComImport, Guid("b63ea76d-1f85-456f-a19c-48159efa858b"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface Items {
        void BindToHandler(IntPtr ctx, ref Guid handler, ref Guid iid, out IntPtr result);
        void GetPropertyStore(uint flags, ref Guid iid, out IntPtr result);
        void GetPropertyDescriptionList(IntPtr key, ref Guid iid, out IntPtr result);
        void GetAttributes(uint flags, uint mask, out uint attributes);
        void GetCount(out uint count);
        void GetItemAt(uint index, out Item item);
    }
    [ComImport, Guid("d57c7288-d4ad-4768-be02-9d969532d960"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface Dialog {
        [PreserveSig] int Show(IntPtr owner);
        void SetFileTypes(uint count, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex=0)] Filter[] filters);
        void SetFileTypeIndex(uint index);
        void GetFileTypeIndex(out uint index);
        void Advise([MarshalAs(UnmanagedType.Interface)] Events events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(uint options);
        void GetOptions(out uint options);
        void SetDefaultFolder(Item folder);
        void SetFolder(Item folder);
        void GetFolder(out Item folder);
        void GetCurrentSelection(out Item item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName(out IntPtr name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void GetResult(out Item item);
        void AddPlace(Item item, uint placement);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
        void Close(int result);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr filter);
        void GetResults(out Items items);
    }
    [ComImport, Guid("e6fdd21a-163f-4975-9c8c-a69f1ba37034"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface Customize {
        void EnableOpenDropDown(uint id);
        void AddMenu(uint id, [MarshalAs(UnmanagedType.LPWStr)] string label);
        void AddPushButton(uint id, [MarshalAs(UnmanagedType.LPWStr)] string label);
    }
    [Guid("973510db-7d7f-452b-8975-74a85828d354"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface Events {
        [PreserveSig] int OnFileOk(Dialog dialog);
        [PreserveSig] int OnFolderChanging(Dialog dialog, Item folder);
        void OnFolderChange(Dialog dialog);
        void OnSelectionChange(Dialog dialog);
        void OnShareViolation(Dialog dialog, Item item, out uint response);
        void OnTypeChange(Dialog dialog);
        void OnOverwrite(Dialog dialog, Item item, out uint response);
    }
    [Guid("36116642-D713-4b97-9B83-7484A9D00433"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface Controls {
        void OnItemSelected(Customize dialog, uint control, uint item);
        void OnButtonClicked(Customize dialog, uint control);
        void OnCheckButtonToggled(Customize dialog, uint control, [MarshalAs(UnmanagedType.Bool)] bool state);
        void OnControlActivating(Customize dialog, uint control);
    }
    [ComVisible(true), ClassInterface(ClassInterfaceType.None)]
    public class Handler : Events, Controls {
        public Dialog Dialog;
        public string Folder;
        public int OnFileOk(Dialog dialog) { return 0; }
        public int OnFolderChanging(Dialog dialog, Item folder) { return 0; }
        public void OnFolderChange(Dialog dialog) {}
        public void OnSelectionChange(Dialog dialog) {}
        public void OnShareViolation(Dialog dialog, Item item, out uint response) { response=0; }
        public void OnTypeChange(Dialog dialog) {}
        public void OnOverwrite(Dialog dialog, Item item, out uint response) { response=0; }
        public void OnItemSelected(Customize dialog, uint control, uint item) {}
        public void OnButtonClicked(Customize dialog, uint control) {
            if(control==100) {
                Item folder; Dialog.GetFolder(out folder);
                Folder=Picker.PathOf(folder); Dialog.Close(0);
            }
        }
        public void OnCheckButtonToggled(Customize dialog, uint control, bool state) {}
        public void OnControlActivating(Customize dialog, uint control) {}
    }
    public static class Picker {
        public static string PathOf(Item item) {
            IntPtr value; item.GetDisplayName(0x80058000, out value);
            try { return Marshal.PtrToStringUni(value); }
            finally { Marshal.FreeCoTaskMem(value); Marshal.ReleaseComObject(item); }
        }
        public static string[] Choose() {
            object instance=Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("dc1c5a9c-e88a-4dde-a5a1-60f82a20aef7")));
            Dialog dialog=(Dialog)instance;
            Handler handler=new Handler { Dialog=dialog };
            uint cookie=0;
            try {
                uint options; dialog.GetOptions(out options);
                dialog.SetOptions(options | 0x200 | 0x40 | 0x1000 | 0x800);
                dialog.SetTitle(Environment.GetEnvironmentVariable("SCMM_PICKER_TITLE") ?? "Select MS files or a folder");
                dialog.SetFileTypes(1,new Filter[] { new Filter { Name="Mass spectrometry files", Pattern="*.mzML;*.mzXML;*.raw" } });
                ((Customize)instance).AddPushButton(100, Environment.GetEnvironmentVariable("SCMM_PICKER_FOLDER") ?? "Select current folder");
                dialog.Advise(handler,out cookie);
                int result=dialog.Show(IntPtr.Zero);
                if(result==unchecked((int)0x800704c7)) return new string[0];
                Marshal.ThrowExceptionForHR(result);
                if(handler.Folder!=null) return new string[] {handler.Folder};
                Items items; dialog.GetResults(out items);
                try {
                    uint count; items.GetCount(out count);
                    List<string> paths=new List<string>();
                    for(uint i=0;i<count;i++) { Item item; items.GetItemAt(i,out item); paths.Add(PathOf(item)); }
                    return paths.ToArray();
                } finally { Marshal.ReleaseComObject(items); }
            } finally {
                if(cookie!=0) dialog.Unadvise(cookie);
                Marshal.ReleaseComObject(instance);
            }
        }
    }
}
"@
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::Out.Write(([ScmmPicker.Picker]::Choose() -join [Environment]::NewLine))
