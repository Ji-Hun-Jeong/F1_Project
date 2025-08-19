import pandas as pd

"""
날씨데이터는 우선 시간을 초로 바꿔놓고 거기서 초랑 비교해서 큰거 나오기 바로 전의 날씨데이터를 모든 car_data에 적용
Compound는 원핫 인코딩
1. laps데이터에서 IsAccurate된 데이터만 랩넘버에 맞는 랩타임이 있으면 car_data에서 데이터를 수집한다.
2. 각 트랙에 맞는 데이터들을 모두 뽑아서 전처리하기전 파일로 만든다. ex) 랩이 바뀔때[모든 스피드의 합, 모든 rpm의 합, 타이어 수명]
3. 위의 데이터들을 이용해서 원하는 피쳐를 뽑아내어 또 파일로 만든다. ex) [스피드의 평균, 스피드의 표준편차, rpm의 평균 등]
4. 위의 데이터들을 이용해서 전처리를 하고 이걸 또 파일로 만든다. ex) 위의 데이터를 어떤 스케일러든 써서 스케일링
5. 결국 완벽하게 피쳐링이 된 데이터만 파일에서 읽어서 모두 램에 올린후 학습시킨다.
"""
"""
    Time: 해당 랩이 완료된 시점의 세션 시간

    Driver: 드라이버의 영문 약자 (예: NOR -> Lando Norris)

    DriverNumber: 드라이버의 차량 번호

    LapTime: 해당 랩을 완료하는 데 걸린 시간 (랩 타임)

    LapNumber: 현재 랩의 번호

    Stint: 피트 스탑(타이어 교체 등) 없이 연속으로 주행한 랩 구간. 피트 스탑을 하면 숫자가 1씩 증가합니다.

    PitOutTime: 피트에서 나온 시간 (피트 아웃으로 시작된 랩일 경우 기록)

    PitInTime: 피트에 들어간 시간 (피트 인으로 끝나는 랩일 경우 기록)

    Sector1Time, Sector2Time, Sector3Time: 트랙의 각 섹터(1, 2, 3)를 통과하는 데 걸린 시간

    Sector1SessionTime, Sector2SessionTime, Sector3SessionTime: 각 섹터를 완료했을 때의 세션 전체 시간

    SpeedI1, SpeedI2: 각 섹터 끝 지점(중간 계측 지점)에서의 순간 속도

    SpeedFL: 결승선(Finish Line)을 통과할 때의 순간 속도

    SpeedST: 트랙에서 가장 빠른 속도를 측정하는 구간(Speed Trap)에서의 순간 속도

    IsPersonalBest: 해당 랩이 드라이버 개인의 최고 기록인지 여부 (TRUE/FALSE)

    Compound: 사용한 타이어의 종류 (예: SOFT, MEDIUM, HARD)

    TyreLife: 해당 타이어로 주행한 랩 수

    FreshTyre: 새 타이어였는지 여부 (TRUE/FALSE)

    Team: 드라이버의 소속 팀 이름

    LapStartTime: 해당 랩이 시작된 시점의 세션 시간

    LapStartDate: 해당 랩이 시작된 날짜와 시간

    TrackStatus: 랩 주행 중 트랙 상태 (예: 1은 정상, 2는 황색기 등)

    Position: 해당 랩을 완료했을 때의 순위

    Deleted: 해당 랩 타임이 삭제되었는지 여부 (트랙 이탈 등의 사유)

    DeletedReason: 랩 타임이 삭제된 이유

    FastF1Generated: FastF1 라이브러리에 의해 생성된 데이터인지 여부

    IsAccurate: 랩 타임 데이터가 정확한지 여부

    랩 시간, 랩 넘버, 
"""