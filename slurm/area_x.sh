#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
. ~/Desktop/Projects/thesis/venv/bin/activate

python ~/Desktop/Projects/thesis/addshare_plus.py f-mnist magnitude
python ~/Desktop/Projects/thesis/addshare_plus_elliptical.py f-mnist magnitude

python ~/Desktop/Projects/thesis/addshare_plus_groups_server.py f-mnist magnitude 3
python ~/Desktop/Projects/thesis/addshare_plus_groups_server.py f-mnist magnitude 5
python ~/Desktop/Projects/thesis/addshare_plus_groups_server.py f-mnist magnitude 10

python ~/Desktop/Projects/thesis/addshare_plus_groups_server_elliptical.py f-mnist magnitude 3
python ~/Desktop/Projects/thesis/addshare_plus_groups_server_elliptical.py f-mnist magnitude 5
python ~/Desktop/Projects/thesis/addshare_plus_groups_server_elliptical.py f-mnist magnitude 10
# ----------------------------------------------------------------------------------------------------------------------

