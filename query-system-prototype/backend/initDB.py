import argparse

from Database import initDB

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_loc', default="./", help="Desired location for database.")
    parser.add_argument('--db_name', default="newDatabase", help="Name of database.")

    args = parser.parse_args()

    initDB(args.db_loc, args.db_name)

