import pandas as pd
import numpy as np
import sys

def view_parquet():
    # pandas 출력 설정 - 생략 없이 전체 내용 표시
    pd.set_option('display.max_colwidth', None)  # 열 내용 생략 방지
    pd.set_option('display.width', None)         # 출력 너비 제한 해제
    pd.set_option('display.max_rows', None)      # 행 생략 방지 (모든 행 표시)
    np.set_printoptions(threshold=np.inf)        # NumPy 배열 생략 방지
    
    # parquet 파일 읽기
    file_path = "test62.parquet"
    df = pd.read_parquet(file_path)
    
    # 기본 정보 출력
    print(f"\n파일: {file_path}")
    print(f"크기: {df.shape[0]}행 × {df.shape[1]}열")
    
    # 열 목록 표시
    print("\n사용 가능한 열:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    
    # 사용자 입력 받기
    try:
        print("\n보고 싶은 열 번호를 쉼표로 구분해서 입력하세요 (예: 0,1,3):")
        cols_input = input().strip()
        if cols_input.lower() == 'all':
            selected_cols = list(df.columns)
        else:
            col_indices = [int(x.strip()) for x in cols_input.split(',')]
            selected_cols = [df.columns[i] for i in col_indices]
        
        print("\n행 범위 지정 방식을 선택하세요:")
        print("1. 처음부터 N개 행 보기")
        print("2. 특정 범위의 행 보기")
        print("3. 모든 행 보기")
        print("4. 조건부 행 보기")
        print("5. 열에 어떤 데이터값 있는지 보기")
        range_option = input().strip()
        
        if range_option == '1':
            print("보고 싶은 행 수를 입력하세요:")
            rows = int(input().strip())
            print(df[selected_cols].head(rows))
        elif range_option == '2':
            print("시작 행 번호를 입력하세요 (0부터 시작):")
            start_row = int(input().strip())
            print("끝 행 번호를 입력하세요:")
            end_row = int(input().strip())
            # 슬라이싱을 사용하여 특정 범위의 행 선택
            with pd.option_context('display.max_rows', None):
                print(df[selected_cols].iloc[start_row:end_row+1])  # end_row를 포함하기 위해 +1
        elif range_option == '3':
            # 모든 행 표시 (생략 없이)
            with pd.option_context('display.max_rows', None):
                print(df[selected_cols])
        elif range_option == '4':
            print("조건을 적용할 열 번호를 입력하세요:")
            filter_col_idx = int(input().strip())
            filter_col = df.columns[filter_col_idx]
            
            print(f"\n적용할 조건을 입력하세요:")
            print("지원되는 연산자: ==, >, <, >=, <=, !=")
            print("예시: '== True' 또는 '> 500' 또는 '!= 0'")
            condition_input = input().strip()
            
            # 조건 파싱
            try:
                # 조건 분리 (연산자와 값)
                if '>=' in condition_input:
                    op, value = '>=', condition_input.split('>=')[1].strip()
                elif '<=' in condition_input:
                    op, value = '<=', condition_input.split('<=')[1].strip()
                elif '==' in condition_input:
                    op, value = '==', condition_input.split('==')[1].strip()
                elif '!=' in condition_input:
                    op, value = '!=', condition_input.split('!=')[1].strip()
                elif '>' in condition_input:
                    op, value = '>', condition_input.split('>')[1].strip()
                elif '<' in condition_input:
                    op, value = '<', condition_input.split('<')[1].strip()
                else:
                    raise ValueError("지원되지 않는 연산자입니다.")
                
                # 값 변환 시도
                try:
                    # 숫자로 변환 시도
                    numeric_value = float(value)
                    value = numeric_value
                except ValueError:
                    # 숫자가 아니면 문자열이나 불리언으로 처리
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    # 그 외는 문자열로 처리
                
                # 조건에 따라 필터링
                if op == '==':
                    filtered_df = df[df[filter_col] == value]
                elif op == '!=':
                    filtered_df = df[df[filter_col] != value]
                elif op == '>':
                    filtered_df = df[df[filter_col] > value]
                elif op == '<':
                    filtered_df = df[df[filter_col] < value]
                elif op == '>=':
                    filtered_df = df[df[filter_col] >= value]
                elif op == '<=':
                    filtered_df = df[df[filter_col] <= value]
                
                print(f"\n'{filter_col}' 열에서 '{condition_input}' 조건을 만족하는 행 (총 {len(filtered_df)}행):")
                print(filtered_df[selected_cols])
                
            except Exception as e:
                print(f"조건 처리 중 오류 발생: {e}")
                print("올바른 형식의 조건을 입력해주세요.")
        elif range_option == '5':
            print("데이터값을 확인할 열 번호를 입력하세요:")
            col_idx = int(input().strip())
            if col_idx >= 0 and col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                
                # 열의 데이터 타입 확인
                col_type = df[col_name].dtype
                print(f"\n'{col_name}' 열의 데이터 타입: {col_type}")
                
                # 고유값 개수 확인
                unique_count = df[col_name].nunique()
                print(f"고유값 개수: {unique_count}")
                
                # 값 분포 확인 (배열 형태 데이터는 다르게 처리)
                if df[col_name].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                    print("\n이 열은 배열/리스트 데이터를 포함하고 있습니다.")
                    print("첫 5개 샘플:")
                    for i, val in enumerate(df[col_name].head(5)):
                        print(f"  {i}: {val}")
                    
                    # 배열 길이 분포 확인
                    lengths = df[col_name].apply(lambda x: len(x) if hasattr(x, '__len__') else 1)
                    print("\n배열 길이 분포:")
                    print(lengths.value_counts().sort_index())
                else:
                    # 일반 데이터는 value_counts로 확인
                    print("\n값 분포:")
                    value_counts = df[col_name].value_counts(dropna=False)
                    
                    # 값이 너무 많으면 상위 20개만 표시
                    if len(value_counts) > 20:
                        print("(상위 20개 값만 표시)")
                        for value, count in value_counts.head(20).items():
                            print(f"{value}: {count}개")
                        print(f"...외 {len(value_counts)-20}개 값이 더 있습니다.")
                    else:
                        for value, count in value_counts.items():
                            print(f"{value}: {count}개")
                    
                    # 숫자 데이터인 경우 통계 정보 추가
                    if np.issubdtype(col_type, np.number):
                        print("\n기초 통계량:")
                        print(df[col_name].describe())
            else:
                print("잘못된 열 번호입니다.")
        else:
            print("잘못된 입력입니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    view_parquet()