from AGB_tables import make_AGB_tables
from SNIa_tables import make_SNIa_tables
from CCSN_tables import make_CCSN_tables
from figures.comparison import make_comparison

if __name__ == "__main__":

    # Make and combine AGB yield tables
    make_AGB_tables()

    # Make SNIa yield tables
    make_SNIa_tables()

    # # Make CCSN yield tables
    make_CCSN_tables()

    # Plot comparison figure among others
    make_comparison()