#%%
cd 



# %%
import subprocess
subprocess.run(["pip3", "install", "earthengine-api", "--quiet"])
subprocess.run(["pip3", "install", "dropbox"])

# %%
import ee
import pandas as pd

# %%
ee.Authenticate()
ee.Initialize(project='grad-thesis-475918')


