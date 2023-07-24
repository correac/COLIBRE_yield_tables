from AGB_tables import make_AGB_tables
from SNIa_tables import make_SNIa_tables
from CCSN_tables import make_CCSN_tables

if __name__ == "__main__":

    # Make and combine AGB yield tables
    make_AGB_tables()


    # Make SNIa yield tables
    # make_SNIa_tables()
    # plot_SNIa_yield_tables()

    # Make CCSN yield tables
    make_CCSN_tables()

    # plot_compare_CCSN_yield_tables()