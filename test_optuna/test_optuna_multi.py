import optuna
from optuna.study import StudyDirection

def objective(trail):
    x = trail.suggest_float('x', -10, 10)
    y = trail.suggest_float('y', -10, 10)
    w = x**2 + y
    z = x*y - 2*y
    if x > 5:
        trail.set_user_attr('x_attr', x)
    if y > 5:
        trail.set_user_attr('y_attr', y)
    return w, z

if __name__ == '__main__':
    study = optuna.create_study(
        storage='sqlite:///db.sqlite3',
        study_name='multi-objective',
        directions=[StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
    )
    study.optimize(objective, n_trials=20)
    print(study.best_trials)