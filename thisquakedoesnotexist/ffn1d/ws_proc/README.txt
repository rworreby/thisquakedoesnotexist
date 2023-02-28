# Relevant script for processing
# Run in this order
python build_training_set.py
python set_attr_table.py
python set_pga.py
python subset_data_vs30.py


        # store normalization constant for conditional variables
        self._set_vc_max()
        # initialize binning configuration
        self._init_bins()
        # set values for conditional variables
        self._init_vcond()
