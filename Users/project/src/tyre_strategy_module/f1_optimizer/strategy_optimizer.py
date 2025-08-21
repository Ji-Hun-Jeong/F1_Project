# f1_optimizer/strategy_optimizer.py
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from .utils import convert_numpy_types

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

def evaluate_strategy(lap_data_df, params_data, strategy_plan, initial_tyre, race_laps):
    # strategy_plan 예시: [{'lap': 15, 'tyre': 'HARD'}, {'lap': 30, 'tyre': 'MEDIUM'}]
    
    total_time = 0.0
    current_compound = initial_tyre
    last_pit_lap = 0
    
    pit_stops = {s['lap']: s['tyre'] for s in strategy_plan}

    for _, row in lap_data_df.iterrows():
        lap_num = int(row['lap'])
        
        # 피트 스톱 수행
        if lap_num in pit_stops:
            total_time += params_data['dry_strategy_params']['pit_stop_time']
            current_compound = pit_stops[lap_num]
            last_pit_lap = lap_num

        stint_length = lap_num - last_pit_lap
        
        # 랩타임 계산 (기존 로직과 거의 동일)
        is_wet = row['rainfall'] > 0
        params_key = 'wet_strategy_params' if is_wet and 'wet_strategy_params' in params_data else 'dry_strategy_params'
        
        if current_compound not in params_data[params_key]['compound_performance']:
             params_key = 'dry_strategy_params' # Fallback
             
        perf = params_data[params_key]['compound_performance'][current_compound]
        degradation = 1 + (perf["degradation_rate"] * stint_length / 100)
        adjusted_laptime = row['laptime'] * perf["base_pace"] * degradation
        total_time += adjusted_laptime
        
    return total_time

def evaluate_hybrid_strategy(individual, lap_data_df, params_data, initial_compound, race_laps):
    """
    GA가 제안한 전략(individual)의 '총 시간'과 '리스크'를 모두 평가합니다.
    """
    MINIMUM_STINT_LAPS = 10

    pit1_lap, pit1_tyre_idx, pit2_lap, pit2_tyre_idx = individual
    plan = []
    if pit1_lap > 0: plan.append({'lap': int(pit1_lap), 'tyre': COMPOUNDS[pit1_tyre_idx]})
    if pit2_lap > 0: plan.append({'lap': int(pit2_lap), 'tyre': COMPOUNDS[pit2_tyre_idx]})
    
    stint_lengths = []
    last_pit_lap = 0
    # 1-스톱 또는 2-스톱 전략의 모든 스틴트 길이를 계산
    pit_laps_sorted = sorted([p['lap'] for p in plan])
    for pit_lap in pit_laps_sorted:
        stint_lengths.append(pit_lap - last_pit_lap)
        last_pit_lap = pit_lap
    stint_lengths.append(race_laps - last_pit_lap)

    # 최소 스틴트 길이 검사 (기존 로직)
    for length in stint_lengths:
        if length < MINIMUM_STINT_LAPS:
            return (999999,)


    # --- ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분을 추가하세요 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ---
    # [핵심 개선] 스틴트 균형 페널티 추가
    balance_penalty = 0
    if len(stint_lengths) > 1: # 피트스톱이 있을 경우에만
        # 스틴트 길이의 표준편차를 계산하여 불균형도를 측정
        stint_std_dev = np.std(stint_lengths)
        # 표준편차가 클수록(불균형할수록) 페널티 증가 (가중치 0.5는 조절 가능)
        balance_penalty = stint_std_dev * 4.0
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 추가하세요 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # 2. F1의 '2가지 컴파운드 사용' 규정 검사
    # (시작 타이어는 INITIAL_COMPOUND 변수를 사용한다고 가정)
    compounds_used = {initial_compound}
    for pit in plan:
        compounds_used.add(pit['tyre'])
    
    # 사용된 컴파운드 종류가 2가지 미만이면 엄청난 페널티를 부여하고 즉시 반환
    if len(compounds_used) < 2 and len(plan) > 0: # 1스톱 이상인데 컴파운드가 2개 미만일 때
        return (999999,)


    # 2. '총 레이스 시간' 계산 (기존과 동일)
    # evaluate_strategy 함수는 이제 '헬퍼' 함수가 됩니다.
    total_time = evaluate_strategy(lap_data_df, params_data, plan, initial_compound, race_laps)

    MAX_RACE_SECONDS = 7200
    if total_time > MAX_RACE_SECONDS:
        # 규정 시간을 초과하는 전략은 최악의 점수를 부여하여 즉시 도태시킴
        return (999999,)

    # 3. '리스크 패널티' 계산 (반응형 모델 인수 사용)
    risk_penalty = 0
    dry_params = params_data['dry_strategy_params'] # 결정용 파라미터 로드
    
    for pit_stop in plan:
        pit_lap = pit_stop['lap']
        
        # 해당 랩의 데이터(ML 예측치) 가져오기
        lap_info = lap_data_df[lap_data_df['lap'] == pit_lap]
        if not lap_info.empty:
            wear_at_pit = lap_info.iloc[0]['tyre_wear']
            proba_at_pit = lap_info.iloc[0]['pit_proba']

            # 패널티 조건 1: 타이어를 너무 한계까지 몰아붙인 전략에 큰 패널티
            if wear_at_pit > dry_params['TyreLife_threshold'] * 1.5: # 임계값의 150% 초과 시
                penalty = (wear_at_pit - dry_params['TyreLife_threshold']) * 5 # 초과한 만큼 큰 패널티
                risk_penalty += penalty
                print(f"DEBUG: Lap {pit_lap} - High Wear Penalty! Wear:{wear_at_pit:.1f} -> Penalty:{penalty:.1f}s")

            # 패널티 조건 2: ML모델이 전혀 피트인을 예측하지 않은 랩에 들어가는 전략에 작은 패널티
            if proba_at_pit < dry_params['pit_proba_threshold'] * 0.1: # 임계값의 10% 미만 시
                penalty = 10 # 10초 고정 패널티
                risk_penalty += penalty
                print(f"DEBUG: Lap {pit_lap} - Low Proba Penalty! Proba:{proba_at_pit:.2f} -> Penalty:{penalty:.1f}s")

    # 최종 적합도 = 총 시간 + 리스크 패널티
    final_score = total_time + risk_penalty + balance_penalty
    return (final_score,) # DEAP는 튜플로 반환해야 함


def create_random_strategy(creator_individual, race_laps):
    num_stops = random.randint(1, 2)
    
    if num_stops == 1:
        pit1_lap = random.randint(10, race_laps - 10)
        pit1_tyre = random.randint(0, 2)
        # ⭐️ 그냥 리스트가 아닌, creator.Individual로 감싸서 반환
        return creator.Individual([pit1_lap, pit1_tyre, 0, 0]) 
    else: # 2 stops
        pit1_lap = random.randint(10, int(race_laps / 2))
        pit2_lap = random.randint(pit1_lap + 10, race_laps - 10)
        pit1_tyre = random.randint(0, 2)
        pit2_tyre = random.randint(0, 2)
        # ⭐️ 여기도 creator.Individual로 감싸서 반환
        return creator.Individual([pit1_lap, pit1_tyre, pit2_lap, pit2_tyre])


def custom_mutate(individual, low, up, indpb):
    # DEAP의 기본 돌연변이 함수를 먼저 실행
    tools.mutUniformInt(individual, low, up, indpb)
    # ⭐️ 핵심 수정: 두 피트스톱 랩이 같아지는 경우를 처리하는 로직 추가 ⭐️
    if individual[2] > 0: # 2스톱 전략일 경우에만 검사
        if individual[0] == individual[2]:
            # 두 랩이 같으면, 비현실적이므로 2스톱을 1스톱으로 변경
            individual[2] = 0
            individual[3] = 0
        elif individual[0] > individual[2]:
            # 순서가 어긋났으면 바로잡음
            individual[0], individual[2] = individual[2], individual[0]
    return individual, # 튜플로 반환해야 함

def custom_mate(ind1, ind2):
    # DEAP의 기본 교배 함수를 먼저 실행
    tools.cxTwoPoint(ind1, ind2)
    # ⭐️ 핵심 수정: 교배 후 생성된 두 자식 전략 모두에 대해 검사 ⭐️
    for ind in [ind1, ind2]:
        if ind[2] > 0: # 2스톱 전략일 경우에만 검사
            if ind[0] == ind[2]:
                # 두 랩이 같으면, 1스톱으로 변경
                ind[2] = 0
                ind[3] = 0
            elif ind[0] > ind[2]:
                # 순서가 어긋났으면 바로잡음
                ind[0], ind[2] = ind[2], ind[0]
    return ind1, ind2

def run_optimization(lap_data_df, params_data, initial_compound, race_laps):
    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_random_strategy, creator.Individual, race_laps=race_laps)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # partial을 사용하여 추가 인자 고정
    from functools import partial
    toolbox.register("evaluate", partial(evaluate_hybrid_strategy, 
                                          lap_data_df=lap_data_df, 
                                          params_data=params_data, 
                                          initial_compound=initial_compound, 
                                          race_laps=race_laps))

    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate, low=[10,0,0,0], up=[race_laps-10, 2, race_laps-10, 2], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    NGEN = 40
    CXPB, MUTPB = 0.5, 0.2
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    result_pop, logbook = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, stats=stats, verbose=True)

    best_individual = tools.selBest(result_pop, k=1)[0]
    
    return format_strategy_result(best_individual, initial_compound, race_laps, params_data['metadata']['track'])

def format_strategy_result(best_individual, initial_compound, race_laps, track_name):
    best_fitness = best_individual.fitness.values[0]
    pit1_lap, pit1_tyre_idx, pit2_lap, pit2_tyre_idx = best_individual

    pit_stops = []
    if pit1_lap > 0:
        pit_stops.append({'lap': int(pit1_lap), 'tyre': COMPOUNDS[pit1_tyre_idx]})
    if pit2_lap > 0:
        pit_stops.append({'lap': int(pit2_lap), 'tyre': COMPOUNDS[pit2_tyre_idx]})
    pit_stops = sorted(pit_stops, key=lambda x: x['lap'])

    strategy_summary = f"Start with {initial_compound} -> "
    if not pit_stops:
        strategy_summary = f"0-Stop: Run entire race on {initial_compound}"
    else:
        for pit in pit_stops:
            strategy_summary += f"Lap {pit['lap']} PIT to {pit['tyre']} -> "
    strategy_summary += "Finish"

    # --- final_result_json 생성 로직 시작 ---
    final_result_json = {}
    final_result_json['track_name'] = track_name
    final_result_json['initial_compound'] = initial_compound
    final_result_json['total_laps'] = race_laps
    final_result_json['estimated_race_time_seconds'] = best_fitness
    final_result_json['estimated_race_time_minutes'] = round(best_fitness / 60, 2)
    final_result_json['total_pit_stops'] = len(pit_stops)
    final_result_json['strategy_summary'] = strategy_summary

    stints_list = []
    current_compound = initial_compound
    stint_start_lap = 1

    if not pit_stops:
        stints_list.append({
            "stint_number": 1, "compound": current_compound, "start_lap": stint_start_lap,
            "end_lap": race_laps, "stint_length": race_laps
        })
    else:
        for i, pit in enumerate(pit_stops):
            end_lap = pit['lap']
            stint_length = end_lap - stint_start_lap + 1
            stints_list.append({
                "stint_number": i + 1, "compound": current_compound, "start_lap": stint_start_lap,
                "end_lap": end_lap, "stint_length": stint_length
            })
            current_compound = pit['tyre']
            stint_start_lap = end_lap + 1
        
        final_stint_length = race_laps - stint_start_lap + 1
        if final_stint_length > 0:
            stints_list.append({
                "stint_number": len(pit_stops) + 1, "compound": current_compound, "start_lap": stint_start_lap,
                "end_lap": race_laps, "stint_length": final_stint_length
            })

    final_result_json['stints'] = stints_list
    # --- final_result_json 생성 로직 끝 ---
    
    return convert_numpy_types(final_result_json)