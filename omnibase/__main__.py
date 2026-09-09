from .cli import main

# Guarded because `sweep --workers N` spawns worker processes, and on Windows a spawned child
# re-imports this module: without the guard it would run the whole CLI again, recursively.
if __name__ == "__main__":
    main()
