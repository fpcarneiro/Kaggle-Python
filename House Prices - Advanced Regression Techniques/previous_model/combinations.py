models = []
models.append(("lasso", model_lasso))
models.append(("ridge", model_ridge))
models.append(("svr", model_svr))
models.append(("ENet", model_ENet))
models.append(("KRR", model_KRR))
models.append(("byr", model_byr))
models.append(("rforest", model_rforest))
models.append(("xgb", model_xgb))
models.append(("GBoost", model_GBoost))
models.append(("lgb", model_lgb))
models.append(("lasso_lars", model_lasso_lars))
models.append(("lsvr", model_lsvr))


import itertools as it
#len(models)+1

qtd_comb = 0
best_score = 1
best_comb = []

for i in range(2, 5):
    print(i)
    combinations = it.combinations(models,i)
    for comb in combinations:
        qtd_comb += 1
        nomes_train = []
        models_train = []
        for nome, model in comb:
            nomes_train.append(nome)
            models_train.append(model)
        print(nomes_train)
        #print(models_train)
        averaged_models = em.AveragingModels(models = models_train)
        score_avg = np.sqrt(pp.score_model(averaged_models, train_X_reduced, train_y)).mean()
        print("Score: {:.6f}\n".format(score_avg.mean()))
        if score_avg < best_score:
            best_score = score_avg
            best_comb = nomes_train[:]
        print()
print(qtd_comb)
print(best_comb)
print(best_score)



['svr', 'KRR']
0.109198528894

Best Tree Combination
['xgb', 'GBoost', 'lgb']
0.112538345478