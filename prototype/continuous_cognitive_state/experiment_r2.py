from .dynamics import DynamicalState


def run() -> None:
    state = DynamicalState()
    print("initial:", state.vector())
    state.tick(drive=1.0)
    print("after stimulus:", state.vector())
    for i in range(1, 11):
        state.tick()
        print(f"internal tick {i}: {state.vector()}")


if __name__ == "__main__":
    run()
