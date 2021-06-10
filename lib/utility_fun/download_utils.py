import requests
import subprocess
import os
from pathlib import Path
import traceback
from tqdm import tqdm

USE_WGET=False

def download_file(url, output_path, cwd=None, use_wget=None, use_cached=False):
    """Fetch url and store it in a file specified by output_path.
    The requests library is used by default, can be configured to use wget by seting use_wget parameter to True
    or similarly a global variable USE_WGET.
    The use_cached==True option allows to skip the download if the file already exists.
    """
    if use_wget is None:
        use_wget = USE_WGET
    print ()
    print (f"Download {url}")
    print (f"      to {output_path}")

    folder = Path(output_path).parent
    try:
        if not folder.exists():
            folder.mkdir()
    except:
        traceback.print_exc()

    if use_cached and os.path.exists(output_path):
        return 

    if use_wget:
        if cwd is None:
            cwd = str(folder)
        output_file = Path(output_path).name
        command = ["wget", url, "-O", output_file]
        print(" ".join(command))
        p = subprocess.call(command ,cwd=cwd)
    else:
        r = requests.get(url, stream=True)
        with open(output_path, 'wb') as fd:
            for chunk in tqdm(r.iter_content(chunk_size=128)):
                fd.write(chunk)

def wgrip2_path():
    "Return path to the wgrib2 executable or raise an exception"
    import shutil
    pp=shutil.which("wgrib2")
    if pp is None:
        for p in [Path.cwd()] + list(Path(__file__).parents):
            for pp in (p / "wgrib2", p / "wgrib2.exe"):
                if pp.exists() and pp.is_file():
                    return str(pp)
        raise Exception("The wgrib2 utility is not found. Please install (e.g. use install_wgrib2.sh)")
    else:
        return pp

def convert_grib2_to_netcdf(grib2_path, netcdf_path):
    "Convert grib2 file to a netcdf file"
    print()
    print (f"Convert     {grib2_path}")
    print (f"  to netcdf {netcdf_path}")
    wgrib2 = wgrip2_path()
    np = Path(netcdf_path)
    cwd = str(np.parent)
    output_file = np.name
    command = [wgrib2, grib2_path, "-append", "-netcdf", output_file]
    print(" ".join(command))
    p = subprocess.call(command ,cwd=cwd)
    

def convert_rain_files(rainfall_path, remove_original_files=False):
    """Convert grib2 rainfall files in the specified directory to netcdf"""
    import os
    from os import listdir
    from os.path import join, isfile

    print(f"Convert rain files in {rainfall_path}")
    rain_files = [f for f in listdir(rainfall_path) if isfile(join(rainfall_path, f))]
#    os.chdir(rainfall_path)                 # TODO Maybe not necessary?
    pattern1='.pgrb2a.0p50.bc_06h'
    pattern2='.pgrb2a.0p50.bc_24h'

    for files in rain_files:
        if pattern2 in files:
            convert_grib2_to_netcdf(files, join(rainfall_path, "rainfall_24.nc"))
#            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_24.nc'%files ,cwd=rainfall_path)
            if remove_original_files:
                os.remove(files)
        if pattern1 in files:
            convert_grib2_to_netcdf(files, join(rainfall_path, "rainfall_06.nc"))
#            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_06.nc'%files ,cwd=rainfall_path)
            if remove_original_files:
                os.remove(files)

if __name__ == "__main__":
    download_file("https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz.v3.0.2", "wgrib2.tgz", use_wget=True)