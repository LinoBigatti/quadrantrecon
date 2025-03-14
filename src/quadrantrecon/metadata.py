import os
import sys
import pathlib
from dateutil.parser import parse as parse_date
#from tabulate import tabulate
#from operator import itemgetter 

# Expected metadata format in path:
# climate/date/location/site/intertidal
def get_metadata_from_path(path: str, starting_path: str):
    path = os.path.relpath(path, starting_path)
    _path = pathlib.Path(path)

    metadata = {"path": path}

    for path_part in reversed(_path.parts):
        path_part = path_part.lower()
        
        # Detect full dates
        try:
            date = parse_date(path_part, fuzzy=False)

            if not "date" in metadata:
                metadata["date"] = date

            continue
        except ValueError:
            pass


        if not "intertidal" in metadata:
            if "alto" in path_part or "alta" in path_part:
                metadata["intertidal"] = "HIGHTIDE"
            if "medio" in path_part or "media" in path_part:
                metadata["intertidal"] = "MIDTIDE"
            if "bajo" in path_part or "baja" in path_part:
                metadata["intertidal"] = "LOWTIDE"
            if "general" in path_part or "pano" in path_part or "planilla" in path_part:
                metadata["intertidal"] = "OTHER"
            
            if "intertidal" in metadata:
                continue

        if not "climate" in metadata:
            if "frio" in path_part:
                metadata["climate"] = "COLDCLIMATE"
            if "calido" in path_part:
                metadata["climate"] = "HOTCLIMATE"
            
            if "climate" in metadata:
                continue
        
        # Site name is directly before the intertidal name
        if not "site" in metadata:
            metadata["site"] = path_part

            continue
        elif "site" in metadata and not "location" in metadata:
            metadata["location"] = path_part

            continue
        
        # Detect dates embedded in strings
        try:
            date = parse_date(path_part, fuzzy=True)

            if not "date" in metadata:
                metadata["date"] = date

                continue
        except ValueError:
            pass

    if not "date" in metadata:
        metadata["date"] = ""
    if not "site" in metadata:
        metadata["site"] = ""
    if not "location" in metadata:
        metadata["location"] = ""
    if not "intertidal" in metadata:
        metadata["intertidal"] = ""
    if not "climate" in metadata:
        metadata["climate"] = ""

    return metadata

#if __name__ == "__main__":
#    starting_path = sys.argv[1]

#    metadatas = []
#    for root, folders, files in os.walk(starting_path):
#        if files and not folders:
#            print("===")
#            metadatas.append(get_metadata_from_path(root, starting_path))


#    print(tabulate(sorted(metadatas, key=itemgetter("location"))))
