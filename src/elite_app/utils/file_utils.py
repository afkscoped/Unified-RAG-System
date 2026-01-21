"""
File Utils - File I/O helpers and utilities
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import shutil
import os
from datetime import datetime


class FileUtils:
    """
    File operation helpers for the training studio
    """
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if not"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> float:
        """Get file size in MB"""
        path = Path(path)
        if path.exists() and path.is_file():
            return path.stat().st_size / (1024 * 1024)
        return 0.0
    
    @staticmethod
    def get_directory_size(path: Union[str, Path]) -> float:
        """Get total directory size in MB"""
        path = Path(path)
        if not path.exists():
            return 0.0
        
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    
    @staticmethod
    def list_files(
        directory: Union[str, Path],
        extensions: List[str] = None,
        recursive: bool = False
    ) -> List[Path]:
        """List files in directory with optional filtering"""
        directory = Path(directory)
        if not directory.exists():
            return []
        
        if recursive:
            files = list(directory.rglob("*"))
        else:
            files = list(directory.glob("*"))
        
        files = [f for f in files if f.is_file()]
        
        if extensions:
            extensions = [e.lower().lstrip('.') for e in extensions]
            files = [f for f in files if f.suffix.lower().lstrip('.') in extensions]
        
        return sorted(files)
    
    @staticmethod
    def list_directories(path: Union[str, Path]) -> List[Path]:
        """List subdirectories"""
        path = Path(path)
        if not path.exists():
            return []
        return sorted([d for d in path.iterdir() if d.is_dir()])
    
    @staticmethod
    def read_json(path: Union[str, Path]) -> Optional[Dict]:
        """Read JSON file"""
        path = Path(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    @staticmethod
    def write_json(path: Union[str, Path], data: Dict, indent: int = 2) -> bool:
        """Write JSON file"""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent)
            return True
        except Exception:
            return False
    
    @staticmethod
    def read_text(path: Union[str, Path]) -> Optional[str]:
        """Read text file"""
        path = Path(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    @staticmethod
    def write_text(path: Union[str, Path], content: str) -> bool:
        """Write text file"""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy file"""
        try:
            src, dst = Path(src), Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False
    
    @staticmethod
    def copy_directory(src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy directory recursively"""
        try:
            src, dst = Path(src), Path(dst)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            return True
        except Exception:
            return False
    
    @staticmethod
    def delete_file(path: Union[str, Path]) -> bool:
        """Delete file"""
        try:
            path = Path(path)
            if path.exists() and path.is_file():
                path.unlink()
            return True
        except Exception:
            return False
    
    @staticmethod
    def delete_directory(path: Union[str, Path]) -> bool:
        """Delete directory recursively"""
        try:
            path = Path(path)
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_file_info(path: Union[str, Path]) -> Optional[Dict]:
        """Get file information"""
        path = Path(path)
        if not path.exists():
            return None
        
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path.absolute()),
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "extension": path.suffix
        }
    
    @staticmethod
    def create_backup(path: Union[str, Path], backup_dir: Union[str, Path] = None) -> Optional[Path]:
        """Create backup of file or directory"""
        path = Path(path)
        if not path.exists():
            return None
        
        if backup_dir is None:
            backup_dir = path.parent / "backups"
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_{timestamp}{path.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            if path.is_file():
                shutil.copy2(path, backup_path)
            else:
                shutil.copytree(path, backup_path)
            return backup_path
        except Exception:
            return None
    
    @staticmethod
    def get_available_space(path: Union[str, Path] = ".") -> float:
        """Get available disk space in GB"""
        path = Path(path)
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(str(path.absolute())),
                    None, None, ctypes.pointer(free_bytes)
                )
                return free_bytes.value / (1024 ** 3)
            else:  # Unix
                stat = os.statvfs(path)
                return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        except Exception:
            return 0.0
