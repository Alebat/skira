import sacred


def get_skira_exp(name):
    ex = sacred.Experiment(name)
    ex.observers.append(sacred.observers.FileStorageObserver(f'runs/{name}'))
    ex.add_config('config.json')
    return ex
