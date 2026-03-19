import argostranslate.package

def mirror_all():
    print("Updating package index...")
    argostranslate.package.update_package_index()
    
    available_packages = argostranslate.package.get_available_packages()
    print(f"Found {len(available_packages)} available language pairs.")

    for pkg in available_packages:
        # Check if already installed
        installed_packages = argostranslate.package.get_installed_packages()
        if pkg in installed_packages:
            print(f"Skipping {pkg.from_code} -> {pkg.to_code} (Already installed)")
            continue
            
        print(f"Downloading {pkg.from_code} -> {pkg.to_code}...")
        try:
            download_path = pkg.download()
            argostranslate.package.install_from_path(download_path)
        except Exception as e:
            print(f"Failed to download {pkg.from_code}: {e}")

    print("\nMirroring complete! Your bot is now 100% offline and multilingual.")

if __name__ == "__main__":
    mirror_all()