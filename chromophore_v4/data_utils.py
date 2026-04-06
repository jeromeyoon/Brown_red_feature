import re
import random
import shutil
from pathlib import Path


def scan_visia_dataset(main_folder: str,
                       max_subjects: int = 1000,
                       mode: str = 'random',
                       seed: int = 42) -> list:
    """
    VISIA 폴더에서 ID 목록을 수집하고 max_subjects개를 선택

    Parameters
    ----------
    main_folder  : 'main_folder/VISIA/brown|red|rgb' 구조의 루트 경로
    max_subjects : 선택할 최대 인원 수
    mode         : 'random' (랜덤 선택) | 'ordered' (정렬 후 앞에서 선택)
    seed         : random 모드 시 재현성 시드

    Returns
    -------
    list of dict: [{'id': str, 'brown': Path, 'red': Path, 'rgb': Path}, ...]
    """
    base       = Path(main_folder) / 'VISIA'
    modalities = ['brown', 'red', 'rgb']

    # rgb 폴더 기준으로 ID 수집 (모든 ID가 3채널 보유 전제)
    rgb_folder = base / 'rgb'
    all_ids = []
    for fpath in rgb_folder.glob('*.PNG'):
        match = re.match(r'^(.+)-c-rgb\.PNG$', fpath.name)
        if match:
            all_ids.append(match.group(1))

    print(f"전체 ID 수: {len(all_ids)}")

    # max_subjects개 선택
    if mode == 'random':
        random.seed(seed)
        selected_ids = random.sample(all_ids, min(max_subjects, len(all_ids)))
        print(f"랜덤 선택 (seed={seed}): {len(selected_ids)}명")
    else:
        selected_ids = sorted(all_ids)[:max_subjects]
        print(f"순서대로 선택: {len(selected_ids)}명")

    # 파일 경로 매핑
    dataset_info = []
    for pid in selected_ids:
        dataset_info.append({
            'id':    pid,
            'brown': base / 'brown' / f'{pid}-c-brown.PNG',
            'red':   base / 'red'   / f'{pid}-c-red.PNG',
            'rgb':   base / 'rgb'   / f'{pid}-c-rgb.PNG',
        })

    return dataset_info


def organize_to_export(main_folder   : str,
                       export_folder : str,
                       max_subjects  : int   = 1000,
                       mode          : str   = 'random',
                       seed          : int   = 42) -> str:
    """
    선택된 1000명 데이터를 export 폴더에 복사하고 tar 압축
    """
    dataset_info = scan_visia_dataset(
        main_folder, max_subjects, mode, seed
    )

    base        = Path(main_folder) / 'VISIA'
    export_base = Path(export_folder) / 'VISIA'
    modalities  = ['brown', 'red', 'rgb']

    # 폴더 생성
    for mod in modalities:
        (export_base / mod).mkdir(parents=True, exist_ok=True)

    # selected_ids.txt 저장 (재현성)
    id_list_path = Path(export_folder) / 'selected_ids.txt'
    with open(id_list_path, 'w') as f:
        f.write('\n'.join(d['id'] for d in dataset_info))
    print(f"ID 목록 저장: {id_list_path}")

    # 파일 복사
    missing = []
    for i, d in enumerate(dataset_info):
        for mod in modalities:
            src = d[mod]
            dst = export_base / mod / src.name
            if src.exists():
                shutil.copy2(src, dst)
            else:
                missing.append(str(src))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dataset_info)} 완료")

    # 결과 요약
    print("\n=== 복사 완료 ===")
    for mod in modalities:
        count = len(list((export_base / mod).glob('*.PNG')))
        print(f"  {mod}: {count}개")
    if missing:
        print(f"⚠️  누락 파일 {len(missing)}개: {missing[:3]}")
    else:
        print("✅ 누락 파일 없음")

    return str(export_base)
