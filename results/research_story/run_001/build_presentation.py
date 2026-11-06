"""Editable Korean PPTX and vector PDF from a shared, checked slide layout.

This script only consumes saved evidence. It never trains or evaluates a model.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from PIL import Image
from fontTools.ttLib import TTFont as FontToolsFont
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).parent
F=json.loads((ROOT/'sources/facts.json').read_text())
REV=F['source_revision']; MAIN=F['main_revision']
BASE='https://github.com/rom-cfd/personal'
W,H=960,540
NAVY='#10243A'; BLUE='#2563EB'; ORANGE='#EA8B23'; TEAL='#087F8C'
RED='#C93D3D'; GRAY='#63758A'; PALE='#EFF4FA'; LINE='#D6E0EA'; WHITE='#FFFFFF'
SLIDES=[]; FIT=[]
OUT=ROOT/'slides'; OUT.mkdir(exist_ok=True)

fonts={
 'regular':(ROOT/'fonts/NanumGothic-Regular.ttf','NanumGothic'),
 'bold':(ROOT/'fonts/NanumGothic-Bold.ttf','NanumGothic'),
 'fallback':(ROOT/'fonts/DejaVuSans.ttf','DejaVu Sans'),
 'fallback_bold':(ROOT/'fonts/DejaVuSans-Bold.ttf','DejaVu Sans')}
CMAPS={}
for name,(path,_) in fonts.items():
    pdfmetrics.registerFont(TTFont(name,str(path)))
    ff=FontToolsFont(str(path)); CMAPS[name]=ff.getBestCmap(); ff.close()


def font_for(ch,bold=False):
    name='bold' if bold else 'regular'
    return name if ord(ch) in CMAPS[name] else ('fallback_bold' if bold else 'fallback')


def width(text,size,bold=False):
    return sum(pdfmetrics.stringWidth(ch,font_for(ch,bold),size) for ch in text)


def wrap(text,w,size,bold=False):
    result=[]
    for para in str(text).split('\n'):
        if not para: result.append(''); continue
        line=''
        for ch in para:
            if width(line+ch,size,bold)>w and line:
                result.append(line.rstrip()); line=ch.lstrip()
            else:line+=ch
        result.append(line.rstrip())
    return result


def rect(s,x,y,w,h,fill=PALE,line=None,r=0):
    s['elements'].append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,line=line,r=r))


def tx(s,text,x,y,w,h,size=16,color=NAVY,bold=False,leading=1.35,link=None):
    lines=wrap(text,w,size,bold)
    needed=size*leading*len(lines)
    FIT.append(dict(slide=len(SLIDES),text=str(text)[:80],height=needed,available=h,passed=needed<=h+.1))
    if needed>h+.1: raise ValueError(f'Slide {len(SLIDES)} overflow {needed:.1f}>{h}: {text}')
    s['elements'].append(dict(kind='text',x=x,y=y,w=w,h=h,text=str(text),lines=lines,size=size,color=color,bold=bold,leading=leading,link=link))


def image(s,name,x,y,w,h):
    path=ROOT/'figures'/f'{name}.png'
    iw,ih=Image.open(path).size
    factor=min(w/iw,h/ih); ww=iw*factor; hh=ih*factor
    s['elements'].append(dict(kind='image',path=str(path),x=x+(w-ww)/2,y=y+(h-hh)/2,w=ww,h=hh))


def new(kicker,title,subtitle='',sources=(),notes=''):
    s=dict(kicker=kicker,title=title,subtitle=subtitle,elements=[],sources=list(sources),notes=notes)
    SLIDES.append(s)
    rect(s,0,0,W,H,WHITE)
    rect(s,0,0,10,H,NAVY)
    tx(s,kicker.upper(),38,21,870,18,size=10,color=TEAL,bold=True)
    tx(s,title,38,50,884,77,size=27,bold=True,leading=1.17)
    if subtitle:tx(s,subtitle,39,117,880,42,size=13,color=GRAY)
    return s


def foot(s,text,color=GRAY):tx(s,text,40,471,880,34,size=11,color=color)


def card(s,x,y,w,h,label,value,body,color=BLUE):
    rect(s,x,y,w,h,PALE,r=8)
    rect(s,x,y,5,h,color,r=2)
    tx(s,label,x+18,y+17,w-34,24,13,GRAY,bold=True)
    tx(s,value,x+18,y+50,w-34,63,36,color,bold=True)
    tx(s,body,x+18,y+120,w-34,h-130,15,leading=1.4)


def bullets(s,items,x,y,w,size=16,gap=18):
    for title,body in items:
        tx(s,title,x,y,w,45,size,bold=True);y+=35
        lines=wrap(body,w,size-1)
        hh=(size-1)*1.4*len(lines)
        tx(s,body,x,y,w,hh+1,size-1,color=GRAY,leading=1.4);y+=hh+gap


def table(s,x,y,w,headers,rows,ratios=None,row_h=38,size=13,header_h=38):
    ratios=ratios or [1/len(headers)]*len(headers)
    xs=[x]
    for r in ratios:xs.append(xs[-1]+w*r)
    rect(s,x,y,w,header_h,NAVY)
    for j,value in enumerate(headers):
        tx(s,value,xs[j]+9,y+9,xs[j+1]-xs[j]-18,header_h-10,size,WHITE,True,leading=1.15)
    yy=y+header_h
    for i,row in enumerate(rows):
        rect(s,x,yy,w,row_h,PALE if i%2==0 else WHITE)
        for j,value in enumerate(row):
            tx(s,str(value),xs[j]+9,yy+9,xs[j+1]-xs[j]-18,row_h-10,size,NAVY,j==0,leading=1.15)
        yy+=row_h
    return yy


def src(path):return f'{BASE}/blob/{REV}/{path}'


def build():
    s=new('Hydrogen kinetics · main → bw','수소 반응계 NN 연구\n무엇을 바꿨고, 무엇을 확인했는가',
          '기준 ANN → 조건부 DeepONet → 물리 벌점 → 전처리 → 시간·다중 스텝 → 일반 ANN',
          ['LONG_COMMITTEE_REPORT.md','results/raw_history_audit/run_001/summary.json','results/ann_multistep_long/run_001/report/test_summary.csv'],
          '이 발표는 main에 있던 100-member ANN 연구를 출발점으로, bw에서 진행한 후속 실험의 질문과 결과를 순서대로 설명합니다. 좋은 그림 하나를 고르는 대신 고정된 18개 테스트 조건에서 물리 보정 없이 얼마나 정확했는지에 초점을 둡니다. 기존 결과와 새 결과를 보존한 커밋을 기준으로 자료를 고정했습니다.')
    rect(s,40,199,878,187,NAVY,r=10)
    tx(s,'핵심 질문',64,218,800,27,15,'#92C8D1',True)
    tx(s,'한 스텝의 작은 오차를 줄이면\n10 ms 자기 예측도 정확해지는가?',64,259,810,109,28,WHITE,True,1.45)
    tx(s,'연구 미팅용 · 본문 23장 + 부록 8장 · 2026-11-06 KST',41,411,870,28,15,GRAY)
    foot(s,'자료 기준: bw ad8c537 · 최신 일반 ANN 장기 결과까지 포함')

    s=new('01 · 먼저 볼 결론','계산의 안정성은 일부 개선됐지만, 넓은 정확도는 아직 부족하다',
          '모든 수치는 저장된 test 결과 기준이며, 테스트 조건은 18개다.',
          ['results/raw_history_audit/run_001/passing_configurations.csv','results/ann_multistep_long/run_001/report/test_summary.csv'],
          '여기서 정확도 통과란 전체 10 ms를 완주하고 온도와 여덟 반응종의 NRMSE가 각각 0.1 이하인 경우입니다. 이는 연구용 사후 진단 문턱이지 CFD 적용 인증이 아닙니다. 과거 raw 기록의 세 통과 설정은 모두 같은 IC39입니다. 이번 일반 ANN 단기·장기 결과에는 새 통과가 없어 고유 조건 수가 늘지 않았습니다. 물리 보정은 별도로 표시하며, 완주만 보고 성공이라고 부르지 않습니다.')
    card(s,40,180,282,258,'보정 없는 정확도 통과 조건','1 / 18','세 모델 설정이 통과했지만\n모두 같은 IC39였다.',BLUE)
    card(s,339,180,282,258,'최신 일반 ANN 실험','0 / 18','단일·다중 스텝, 단기·장기\n모두 정확도 통과는 없었다.',RED)
    card(s,638,180,282,258,'결과를 읽는 원칙','두 지표','완주 수와 정확도 통과 수,\nraw와 repaired를 구분한다.',TEAL)
    foot(s,'정확도 통과: full 10 ms + T·8반응종 모두 NRMSE ≤ 0.1. N2는 별도 점검했다.')

    s=new('02 · 연구 흐름','한 번에 다 바꾸지 않고, 실패 원인을 나누어 확인했다',
          '왜 했는가 → 무엇을 바꿨는가 → 무엇이 남았는가',
          ['results/research_story/run_001/sources/commit_history_complete.json'],
          'main에는 이미 장기 궤적 데이터와 100-member ANN이 있었습니다. bw에서는 먼저 압력·온도·당량비를 나누어 넣는 구조를 비교했습니다. 다음으로 학습을 늘리고 원소 보존 벌점을 넣었으나 넓은 raw 정확도를 얻지 못했습니다. 이후 벌점을 빼고 Box–Cox 지수를 조정했습니다. 그래도 자기 예측 누적 오차가 남아 시간 간격과 다중 스텝 학습을 비교했고, 마지막에는 다중 스텝 효과가 일반 ANN에도 있는지 별도 대조군으로 확인했습니다.')
    steps=[('main 기준','ANN 10/100 평균','앙상블 크기만으로 해결되지 않음'),
           ('bw: 구조','P → T/P → T/P/phi','조건 분리·혼합과 장기 학습 비교'),
           ('bw: 손실','원소 보존 soft constraint','λ를 늘려도 raw 정확도 확대 실패'),
           ('bw: 전처리','Box–Cox γ 6개','온도 점수와 화학종 점수의 차이 확인'),
           ('bw: 시간 학습','간격 τ + 다중 스텝 K','학습과 실제 자기 예측의 차이 축소 시도'),
           ('bw: 일반 ANN','K1 대 K1→64','DeepONet과 분리해 효과 재확인')]
    for i,(label,title,body) in enumerate(steps):
        x=40+(i%3)*298;y=174+(i//3)*140
        rect(s,x,y,282,122,PALE,r=7)
        tx(s,label,x+15,y+13,252,20,11,TEAL,True)
        tx(s,title,x+15,y+40,252,28,18,bold=True)
        tx(s,body,x+15,y+79,252,36,12,color=GRAY)
    foot(s,'main의 연구를 bw의 신규 성과로 합산하지 않았다. 단기·장기·재사용 체크포인트도 구분했다.')

    s=new('03 · 공통 문제 설정','NN은 반응률이 아니라, 정규화된 상태의 증분을 예측한다',
          'Cantera의 정압·단열 H2/air 반응계를 2 us 간격으로 전진시킨다.',
          ['ANN_DEEPONET_WORKFLOW.md','LONG_COMMITTEE_REPORT.md','results/ann_multistep/run_001/config.json'],
          '물리 상태는 온도와 9개 질량분율로 구성되고 압력이 입력에 추가됩니다. 질량분율의 Box–Cox 변환과 온도 스케일링 후 표준화한 좌표에서 다음 상태와 현재 상태의 차이를 학습합니다. 기존 ANN도 이 증분 방식입니다. 학습 데이터는 각 시각의 상태이며, 저장된 바로 다음 상태와의 차이로 정답 증분을 계산합니다. 실제 rollout은 처음만 Cantera 초기 상태를 받고, 이후에는 NN이 만든 상태를 계속 되먹여 5,000번 전진합니다.')
    table(s,40,176,880,['항목','공통 기준'],[
        ['대상','h2o2.yaml 기반 H2/air · 9화학종 · 28반응'],
        ['조건','90 IC: T 900–1600 K · P 1–5 atm · phi 0.5–2'],
        ['상태 / 입력','T + Y9 / 현재 상태 + 압력 (일반 ANN 입력 11개)'],
        ['출력','Δz = 표준화된 다음 상태 − 표준화된 현재 상태'],
        ['실제 평가','2 us × 5,000회 자기 예측 = 10 ms']
    ],ratios=[.18,.82],row_h=47,size=15)
    foot(s,'한 번의 상태 정확도와, 5,000회 되먹인 궤적의 정확도는 다른 문제다.',TEAL)

    s=new('04 · 데이터와 전처리','학습 시작점은 줄였지만, 정답 간격은 2 us로 유지했다',
          'trajectory 단위 train 54 / validation 18 / test 18 분할을 고정했다.',
          ['results/long10ms_data/selected_inputs.csv','results/long10ms_data/sampling_summary.csv','LONG_COMMITTEE_REPORT.md'],
          '이 데이터 구성은 main의 기준 연구에서 이어받은 것입니다. 270,000개 후보 중 8,815개 입력 시작점을 선택합니다. 각 train 궤적의 비평형 지점은 최대 320개, 평형 앵커는 최대 8개입니다. 초반 지점과 시간·상태 변화량을 고려하므로 느린 점화 유도 구간도 보존합니다. 학습 지점을 건너뛰어 골라도 정답은 항상 dense 시계열의 2 us 뒤 상태이며, 다음 선택 지점이 아닙니다. 평균과 표준편차는 선택된 train 현재·다음 상태만으로 맞추고 test로 정규화를 맞추지 않습니다.')
    image(s,'01_sampling',37,174,573,266)
    bullets(s,[('평형 과대표집 완화','선택된 평형 앵커는 376개 / 8,815개.\n조밀한 원자료의 평형 비중은 약80.5%였다.'),
               ('변환 후 train 통계로 표준화','T/1000, Y^γ/γ, log(P)를 사용한다.\nγ 실험에서도 분할과 시작점은 유지했다.')],635,180,277,size=15,gap=22)
    foot(s,'학습 입력을 고르는 것과 rollout 시간 간격을 바꾸는 것은 구분했다.')

    s=new('05 · main에서의 출발점','ANN을 10개에서 100개로 늘려도 오차가 줄지 않았다',
          '이 결과는 main에 이미 있던 물리 보정 포함 앙상블 비교다.',
          ['results/committee100_long10ms/summary.csv','LONG_COMMITTEE_REPORT.md'],
          'main에서 100개 ANN 학습과 평가가 이미 이루어졌고, bw에서의 후속 연구는 이 결과를 출발점으로 삼았습니다. 10개 평균은 원래 100개 중 앞의 10개를 사용하는 부분집합 비교입니다. test 10 ms에서 둘 다 18개를 완주했지만 온도 RMSE는 10개 평균이 약179 K, 100개 평균이 약275 K였습니다. 이 그림은 물리 보정을 포함하므로 보정 없는 100개 평균의 성능을 뜻하지 않습니다. 따라서 모델 수를 늘리는 대신 조건을 넣는 방식과 학습 목표를 바꾸기 시작했습니다.')
    image(s,'02_main_committee',40,176,880,268)
    foot(s,'결정: 앙상블 크기를 키우는 대신 입력 구조·학습 목표·전처리를 차례로 살펴본다.',TEAL)

    s=new('06 · bw의 첫 확장','압력·온도·당량비를 분리해서 넣는 DeepONet을 비교했다',
          '모두 같은 현재 상태에서 다음 상태 증분을 만드는 유한 차원 조건부 모델이다.',
          ['ANN_DEEPONET_WORKFLOW.md','TP_PHI_WORKFLOW.md','P_ONLY_REBUILD_WORKFLOW.md'],
          'P-only는 현재 온도와 질량분율을 branch에 넣고 압력을 trunk에 넣습니다. T/P는 질량분율을 branch, 온도와 압력을 trunk에 둡니다. T/P/phi는 여기에 초기 당량비를 trunk 조건으로 추가하며 rollout 동안 그 초기 phi를 고정합니다. 이 세 구조는 다음 상태 증분을 생성하며 전체 시간장을 한 번에 만들지 않습니다. 혼합 결과는 ANN과 DeepONet을 직렬로 연결하는 것이 아니라, 같은 현재 상태로부터 나온 10개와 10개 후보의 평균입니다. 표의 모델 수와 평가 조합 수를 혼동하면 안 됩니다.')
    table(s,40,175,880,['모델','Branch 입력','Trunk 입력','파라미터'],[
        ['일반 ANN','현재 T·Y9 + P를 MLP에 함께 입력','없음','9,738'],
        ['P-only DeepONet','현재 T·Y9','현재 log(P)','9,719'],
        ['T/P DeepONet','현재 Y9','현재 T, log(P)','9,687'],
        ['T/P/phi DeepONet','현재 Y9','현재 T, log(P), 초기 phi','9,719'],
        ['ANN + Deep 혼합','같은 현재 상태 → ANN10 + Deep10','20개 후보 평균','별도 학습 없음']
    ],ratios=[.22,.31,.32,.15],row_h=48,size=13)
    foot(s,'DeepONet이라는 구조 이름만으로 임의 함수·새 반응계에 대한 일반화가 검증된 것은 아니다.')

    s=new('07 · 구조와 장기 학습','학습을 늘리고 phi를 넣어도, raw 정확도는 일부 조건에 머물렀다',
          '아래는 초기 구조 실험의 long 결과다. 이후 γ 실험과 중복 합산하지 않는다.',
          ['results/raw_history_audit/run_001/summary.json','DEEPONET_LONG_WORKFLOW.md','TP_PHI_WORKFLOW.md'],
          '첫 T/P 비교 후 저장된 P-only 모델이 없어 P-only 10개를 다시 만들고, P-only와 T/P에 추가 학습을 수행했습니다. 그 뒤 T/P/phi도 단기·장기로 비교했습니다. 장기는 당시 연구 전체 최대 6시간 예산이었으며 모델 하나당 6시간이 아닙니다. raw T/P 장기 12개 완주 중 정확도 기준을 통과한 것은 IC39 하나뿐입니다. phi를 추가한 구조와 ANN 혼합은 이 원자료 기준에서 정확도 통과를 추가하지 못했습니다. 최적화 부족 외에 목표 손실과 입력 변환도 살펴볼 이유가 생겼습니다.')
    table(s,40,174,880,['장기 평가 설정','raw 완주 / 18','raw 정확도 통과 / 18'],[
        ['ANN10 (기준 모델 재사용)','2','0'],['P-only10','6','0'],['ANN10 + P-only10','9','0'],
        ['T/P10','12','1  (IC39)'],['ANN10 + T/P10','6','0'],
        ['T/P/phi10','7','0'],['ANN10 + T/P/phi10','11','0']
    ],ratios=[.5,.22,.28],row_h=34,size=13,header_h=37)
    foot(s,'다음 질문: MSE에 원소 보존 벌점을 더하면 자기 예측의 이탈을 줄일 수 있을까?',TEAL)

    s=new('08 · 학습 손실 변경','Soft constraint는 학습 벌점이고, rollout 물리 보정과 다르다',
          '실제 구현은 다음 상태의 원소량 오차에 대한 L1 벌점이다.',
          ['kinetics/soft_constraint_training.py','SOFT_CONSTRAINT_WORKFLOW.md'],
          '업로드된 methane step2의 아이디어를 참고했지만 이 저장소의 실제 구현은 반응률 직접 회귀나 ODE 잔차가 아닙니다. 정규화 증분 MSE에, 예측 다음 질량분율과 Cantera 다음 질량분율의 차이를 원소 질량 행렬로 투영한 절댓값의 합을 더합니다. A는 H/O/N의 3×9 행렬이며 엔탈피 벌점이나 dY/dt로 나누는 항은 없습니다. loss 내부에서는 역변환의 signed 확장을 사용하지만 실제 예측 시 역변환 영역 검사를 생략한 것은 아닙니다. 물리 보정은 학습이 끝난 후 각 rollout 단계에 적용하는 별도 시스템입니다.')
    rect(s,40,180,880,90,NAVY,r=8)
    tx(s,'Loss = MSE(Δz_pred, Δz_true) + λ · mean[ Σ |A · (Y_pred − Y_true,next)| ]',59,206,840,50,19,WHITE,True)
    table(s,40,295,880,['구분','학습 soft constraint','평가 repaired'],[
        ['작동 시점','역전파할 손실에 추가','예측한 후보 상태를 매 스텝 수정'],
        ['물리 정보','H/O/N 원소량 차이','비음수·원소 투영·엔탈피 온도 보정'],
        ['이번 비교','λ=1, 10, 100, 1000','raw와 별도 평가·별도 표기']
    ],ratios=[.16,.42,.42],row_h=45,size=14)
    foot(s,'신경망 출력은 계속 상태 증분이다. 반응률을 직접 출력하도록 바꾼 실험은 아니다.')

    s=new('09 · 벌점 가중치 실험','λ를 키우거나 더 오래 학습해도 raw 정확도 통과는 늘지 않았다',
          'γ=.1 고정 · 4개 학습 계열 + 3개 혼합 평가 · short=10,000 update',
          ['results/penalty_weight_sweep/run_001/FINAL_ARTIFACTS.json','results/element_soft_long/run_001/FINAL_ARTIFACTS.json','results/weight_long_study/run_001/FINAL_ARTIFACTS.json'],
          '먼저 λ10으로 ANN, P-only, T/P, T/P/phi 각10개를 학습하고 장기 연장했습니다. 이후 λ1,100,1000도 같은 순서로 비교했습니다. 그림은 업데이트 수를 맞춘 단기 결과이며 각 칸은 정확도가 아니라 완주 수입니다. raw 정확도 동시 통과는 모든 새 soft 모델에서 0개였습니다. 장기 λ10의 T/P와 phi는 일부 완주가 늘었지만 정확한 케이스를 추가하지 못했고, λ1/100/1000 장기도 정확도 통과가 없었습니다. 당시 장기 실험은 연구 전체6시간으로 가중치별 모델 수가 달라 per-model 예산까지 동일한 비교는 아닙니다. 이후 λ0 MSE로 돌아가 전처리만 분리해서 바꿨습니다. λ0을 보편적인 최적값이라고 증명한 것은 아닙니다.')
    image(s,'03_lambda_short',38,174,600,274)
    bullets(s,[('추가 학습도 수행','λ=10: 40개 연장\nλ=1/100/1000: 120개 연장'),('관찰과 다음 결정','새 soft 모델의 raw 정확도 통과 0.\n이후 soft 항을 빼고 γ만 바꿨다.')],662,181,248,size=15,gap=24)
    foot(s,'큰 λ는 이 설정에서 완주도 악화시켰다. 장기 시간상한 종료를 평탄 도달로 부르지 않았다.')

    s=new('10 · Box–Cox 전처리','γ에 따라 성능이 달랐지만, 보정 없는 정확도는 넓어지지 않았다',
          'γ=.05/.1/.2/.3/.5/1 · 4계열×10개 · MSE만 사용',
          ['results/gamma_long/run_001/FINAL_ARTIFACTS.json','GAMMA_LONG_WORKFLOW.md'],
          'Box–Cox 지수는 미량 화학종과 큰 질량분율의 상대적인 학습 스케일을 바꿉니다. 6개 gamma의 short 기준을 10k로 맞추고 각 모델을 추가 활성150초 학습했습니다. 240개 중20개 gamma.1 ANN/T-P는 Adam상태가 없어 기준10k 재생성 비용도 그150초 안에 포함했고 나머지220개는 정확히 이어받았습니다. 모두 시간상한 종료이며 평탄 도달은0입니다. 여기서 보정 없는 정확도 통과는 P-only gamma.2 IC39 한 조건입니다. 보정 포함 결과도 최대3/18이므로 넓은 성공이라고 말할 수 없습니다. 온도 기반으로 validation gamma를 선택한 결과는 부록에 별도로 명시했습니다.')
    image(s,'06_gamma_accuracy',37,174,884,287)
    foot(s,'각 칸은 joint9 정확도 통과 수 / 18 (완주 수가 아님). long: 평탄 0 / 시간상한 240.',TEAL)

    s=new('11 · 평가 해석에서 배운 점','온도 NRMSE가 작아도, 반응종이 잘 맞는 것은 아니었다',
          'IC07 · γ=.2 · ANN+P 장기 · 물리 보정 포함 예측',
          ['results/research_story/run_001/sources/selected_trace_provenance.json','results/research_story/run_001/sources/facts.json'],
          '예전에 좋아 보인 IC07을 원자료에서 다시 그렸습니다. 이 조건의 Cantera 온도는10ms 동안 약0.0068K만 변합니다. NN은 약2.02K 변해서 같은 축에서는 Cantera가 평평해 보입니다. RMS 정규화는 절대온도 약930K가 분모이므로 T NRMSE0.00161만 보고 성공이라고 하기 쉽습니다. 그러나 H2O와 OH의 NRMSE는 각각781과754에 가깝습니다. 그래서 모든 반응종, 초기 점화 구간, 물리 오차를 함께 보는 평가 기준을 정리했습니다. 그래프가 평평한 원인은 모델이 훌륭해서가 아니라 실제 반응 변화가 작고 축이 그 변화를 가렸기 때문입니다.')
    image(s,'07_metric_trap',40,176,880,260)
    foot(s,'Cantera ΔT≈0.0068 K, NN ΔT≈2.02 K. T NRMSE≈0.00161, H2O≈781, OH≈754.',RED)

    s=new('12 · 보정 없는 모델과 보정 시스템','물리 보정은 완주를 도와주지만, 반응 속도까지 보장하지 않는다',
          'raw = 모델 후보를 그대로 사용 / repaired = 아래 절차를 포함한 시스템 결과',
          ['kinetics/temporal_pilot.py','results/ann_multistep_long/run_001/verification.json'],
          '최신 repaired에서는 역변환 영역 clipping 후 비음수 원소 투영과 엔탈피 기반 온도 복원을 적용합니다. 앙상블에서는 유효 후보만 평균한 뒤 평균 상태도 보정합니다. 이는 chemical ODE를 Cantera로 대신 풀어주는 fallback이 아니며, 열역학 온도 복원과 화학 적분을 혼동하면 안 됩니다. raw는 clipping이나 후보 제외를 사용하지 않아 후보 하나라도 무효이면 평균 rollout이 실패할 수 있습니다. 따라서 repaired의 높은 완주 수는 모델 단독 성능이 아니라 이 보정 시스템을 포함한 성능입니다. 보존식을 맞추더라도 반응 속도나 점화 시각은 틀릴 수 있습니다. 초기 main의 단순 repair와 최신 repaired는 구현이 같다고 가정하지 않았습니다.')
    items=[('1','역변환 영역 clipping'),('2','비음수 · 원소량 투영'),('3','엔탈피로 온도 복원'),('4','유효 후보 평균 후 재보정')]
    for i,(n,label) in enumerate(items):
        x=40+i*223
        rect(s,x,193,207,102,PALE,r=7)
        tx(s,n,x+15,204,177,37,25,TEAL,True)
        tx(s,label,x+15,251,177,34,14,bold=True)
    bullets(s,[('완주가 의미하는 것','10 ms까지 유효한 상태로 계산이 이어졌다는 뜻이다.'),('완주만으로 말할 수 없는 것','Cantera와의 정확도, 점화 시각, 미량 화학종 정확도를 보장하지 않는다.')],45,325,850,size=16,gap=14)

    s=new('13 · 평가 기준과 과거 raw 감사','보정 없는 통과 설정은 세 개였고, 고유 조건은 IC39 하나였다',
          '저장된 10 ms test raw 기록을 감사하고 최신 일반 ANN 결과를 추가 확인했다.',
          ['results/raw_history_audit/run_001/summary.json','results/raw_history_audit/run_001/passing_configurations.csv','results/ann_multistep_long/run_001/report/test_summary.csv'],
          'NRMSE는 사용자 util.py의 RMSE/RMS 기준으로 통일했습니다. 전체10ms를 못 채운 결과에는 full-horizon 점수를 주지 않습니다. joint9는 T와 8반응종을 각각0.1이하로 요구하며 N2는 별도 점검합니다. 감사에서 발견한 세 통과 설정은 T/P10 gamma.1 장기, P-only10 gamma.2 장기, 최근 P-only single-step3개평균 gamma.2 장기이며 모두IC39입니다. 초기0.1ms에서도 통과하는 것은 세 개 중 두 개뿐입니다. 감사는 저장된 held-out rollout에 대한 것이므로 raw 평가가 없는 모든 체크포인트나 corrected100-member의 raw 성능을 포함한다고 말하지 않습니다. 새 일반 ANN은 추가 성공이 없었습니다.')
    table(s,40,178,880,['raw 설정 (모두 IC39)','max NRMSE 10 ms','초기 0.1 ms max'],[
        ['T/P Deep10 · γ=.1 · 장기','0.0743','0.0975'],
        ['P-only Deep10 · γ=.2 · 장기','0.0759','0.1347  (문턱 초과)'],
        ['P-only Deep3 · γ=.2 · 새 장기 대조','0.0861','0.0558']
    ],ratios=[.5,.24,.26],row_h=51,size=14)
    tx(s,'NRMSE = RMSE(Cantera, NN) / RMS(Cantera)',47,395,870,30,21,BLUE,True)
    foot(s,'무차원 비율이며 %가 아니다. N2를 포함해도 세 통과 설정은 동일하다. 정확도 문턱은 사후 진단용이다.')

    s=new('14 · 실제 궤적 확인','IC39의 점화 구간은 맞췄지만, 다른 17조건으로 확장되지 않았다',
          'T0=1250.05 K · P0=2.297 atm · phi0=.646 · 보정 없는 세 설정',
          ['results/raw_history_audit/run_001/passing_configurations.csv','results/research_story/run_001/sources/selected_trace_provenance.json'],
          '대표 성공 그림은 전체 성능을 대표하지 않습니다. IC39에서는 세 모델이 온도와 주요 라디칼의 점화 변화를 따라갑니다. 그래도 질량분율 합의 최대 이탈이 약0.0064~0.0129이고 원소량 오차도 남아 있습니다. NRMSE 정확도 문턱 통과가 물리 보존의 증명이 아님을 보여줍니다. full10ms에서는 통과하지만 P-only10 gamma.2의 초기0.1ms 최악 오차는0.1347로 더 큽니다. 이에 시간구간별 오차와 conservation을 같이 보기로 했습니다.')
    image(s,'09_IC39_three_raw',37,159,885,305)
    foot(s,'각 패널은 0–0.15 ms 확대. 세 설정 모두 질량·원소량 drift가 남아 있어 물리 보존 인증은 아니다.')

    s=new('15 · 시간 정보를 쓰는 두 접근','시간 간격을 바꾸는 모델과, 여러 번 되먹이며 학습하는 모델을 나눴다',
          '둘 다 기존 DeepONet 기반으로 시작했으며, 모든 최종 평가는 2 us 고정이다.',
          ['results/temporal_pilot/run_001/config.json','kinetics/temporal_pilot.py'],
          '시간 간격 모델은 초기조건에서 전체 시간장을 한 번에 만드는 방식이 아닙니다. branch에 현재 상태와 압력을 넣고 trunk에 앞으로의 간격tau를 넣어 그 뒤 상태의 증분을 만듭니다. 기존dense정답에서2,4,8,16us를 만들고 배치 절반은2us, 나머지는균등하게 둡니다. 시작점8815개에서 유효35098쌍이 만들어져 새Cantera궤적은 필요하지 않았습니다. 다중 스텝 모델은 기존2us예측을 K번 되먹이고 1~K모든 중간오차를 평균해 역전파합니다. 시간4분의1씩K1,4,16,64로 늘렸고 장기에서는K64고정입니다. K는 예측간격이 아니라 학습에서 이어가는 횟수입니다.')
    table(s,40,174,880,['접근','바꾼 것','추가 정답 / 실제 평가'],[
        ['시간 간격 DeepONet','trunk = 앞으로의 간격 tau','2/4/8/16 us 정답 / 평가는 2 us'],
        ['다중 스텝 DeepONet','2 us 예측을 K번 되먹인 loss','모든 중간 오차 / 평가는 2 us']
    ],ratios=[.25,.35,.4],row_h=51,size=14)
    image(s,'16_K_curriculum',48,330,864,113)
    foot(s,'기존 dense 정답을 재사용했다. 전체 시간장을 직접 출력하는 모델은 이번에 실행하지 않았다.')

    s=new('16 · 시간·다중 스텝 DeepONet 결과','다중 스텝은 raw 완주를 늘렸지만, 새 정확도 통과는 없었다',
          'seed11/22/33 각각 학습 · 아래는 3개 평균 · γ=.2 · 단기150초 + 장기최대150초/모델',
          ['results/temporal_pilot/run_001/report/test_summary.csv','results/temporal_pilot_long/run_001/report/test_summary.csv','results/temporal_pilot_long/run_001/training.csv'],
          '단기150초는 앞선10k기준과 다른 시간예산입니다. 모델을 처음부터150초 학습하고 가중치와Adam을 이어서 최대150초 더 학습했습니다. 그래프는 seed별이 아니라mean3입니다. raw다중스텝 완주는14→13이고 정확도통과는0입니다. 시간간격 모델은0→0이고, 기존single-step 대조군만7→9완주와0→1정확도통과를 얻었습니다. repaired에서는 세방식모두18완주이지만 정확도는대조군3→2, 새두방식0입니다. interval은 장기3seed모두 loss평탄 기준에 도달했는데도 rollout은 실패했습니다. 학습평탄과장기정확도가 별개라는 증거입니다.')
    image(s,'10_deeponet_temporal',40,176,880,269)
    foot(s,'시간 간격: 평탄 3/3이지만 raw 완주 0. repaired 정확도: 대조군 3→2, 새 두 방식은 0.')

    s=new('17 · 일반 ANN으로 재확인','다중 스텝 효과를 일반 ANN에서도 별도 대조 실험했다',
          'branch/trunk 없는 11→64→64→64→10 tanh MLP · 9,738 parameters',
          ['results/ann_multistep/run_001/config.json','results/ann_multistep_long/run_001/config.json','results/ann_multistep_long/run_001/training.csv'],
          '사용자가 ANN으로도 단기·장기를 비교하도록 요청해 일반MLP를 별도 학습했습니다. 앞선다중스텝DeepONet 결과와혼동하면 안 됩니다. 같은seed에서single과multi의초기가중치를같게 만들고, 데이터분할·정규화·gamma.2·배치512·Adam을 고정했습니다. 각seed각방식단기150초, 정확한Adam복원후최대150초추가입니다. 학습softconstraint나학습중물리보정은없습니다. K가 커지면 한 update 비용이 커지므로 시간예산을 맞췄지 update수를맞춘실험은 아닙니다. 단기게시와원격SHA확인gate가끝난후 장기를시작했습니다.')
    table(s,40,176,880,['고정한 것','비교한 것'],[
        ['같은 seed 초기 가중치 · seed 11 / 22 / 33','단일 스텝 loss: K=1'],
        ['같은 8,815개 시작점 · gamma=.2 · 배치512','다중 스텝 loss: K1→4→16→64'],
        ['각 모델 단기150초 + 추가 최대150초','장기에서는 다중 스텝 K64 유지'],
        ['가중치 + Adam 정확 복원','raw / repaired, seed별 / mean3 각각 평가']
    ],ratios=[.54,.46],row_h=54,size=15)
    foot(s,'이번 “단기”는 10,000 update가 아니라 활성 학습 150초다. 동시간 비교에서 update 수는 달라진다.',TEAL)

    s=new('18 · 일반 ANN 단기·장기 결과','일반 ANN도 다중 스텝에서 일부 더 완주했지만, 정확하지 않았다',
          '3개 평균 · raw 결과 먼저 · 정확도 동시 통과는 seed별로도 모두 0',
          ['results/ann_multistep/run_001/report/test_summary.csv','results/ann_multistep_long/run_001/report/test_summary.csv','results/ann_multistep_long/run_001/report/short_long_common_summary.csv'],
          'raw single-step mean3는단기장기모두0완주, multi는1→2완주입니다. repaired에서는single18→18, multi13→14완주하지만 양쪽모두정확도통과0이며개별seed도0입니다. raw공통완주IC74에서worst9는62.688→57.656으로줄었으나문턱0.1보다훨씬큽니다. repaired공통집합을쓰면 single의worst9중앙값은6.431→4.702, multi는23.411→23.281입니다. 서로다른완주집합의중앙값을단순비교하지않았습니다. 이결과는완주향상과정확도향상이같지않음을다시보여줍니다.')
    image(s,'11_ANN_results',40,176,880,269)
    foot(s,'repaired 완주: 단일 18→18, 다중 13→14. 두 방식의 정확도 통과는 단기·장기 모두 0.',RED)

    s=new('19 · 학습과 seed 편차','학습 loss의 평탄과, 예측 결과의 정확도는 일치하지 않았다',
          '일반 ANN 다중 스텝 · 고정 K64 training probe와 seed별 raw 완주',
          ['results/ann_multistep_long/run_001/training.csv','results/ann_multistep_long/run_001/models/multistep_seed11/history.csv','results/ann_multistep_long/run_001/report/test_summary.csv'],
          '단기다중스텝에서는K가바뀌므로서로다른K의loss를하나의수렴곡선으로해석하지않습니다. 장기에서는K64로고정한probe를보았습니다. seed11은최종LR1e-6에서평탄기준을만족했지만정확도통과는0입니다. seed22와33은추가시간상한이었습니다. raw장기완주는seed별0,9,7이지만mean3는2이므로평균앙상블이항상더안정적이지도않습니다. test에서제일좋은seed를골라최종모델이라고주장하지않고seed편차를보고합니다. 평탄기준은5활성초간격probe,누적0.5%개선없는3회마다LR0.3배,1e-6에서6회추가무개선입니다.')
    image(s,'13_ANN_K64_probe',35,166,550,274)
    table(s,612,187,306,['장기 raw','완주 / 18'],[['seed11','0'],['seed22','9'],['seed33','7'],['mean3','2']],ratios=[.6,.4],row_h=46,size=15)
    foot(s,'일반 ANN 장기 종료: 평탄 1 / 시간상한 5. “loss 평탄”은 “10 ms가 정확함”을 뜻하지 않는다.')

    s=new('20 · 최신 같은 시간 예산 비교','이 설정에서는 DeepONet 다중 스텝이 ANN보다 더 많이 완주했다',
          'γ=.2 · 각 3seed 평균 · 단기150초 + 장기최대150초 · 같은 test18',
          ['results/ann_multistep_long/run_001/report/historical_deeponet_comparison.csv','results/temporal_pilot_long/run_001/report/test_summary.csv'],
          '장기mean3를보면raw완주는P-onlysingle9,multi13,일반ANNsingle0,multi2입니다. 다만정확도통과는DeepPsingle의IC39하나이며다중스텝두구조는모두0입니다. repaired에서는완주가18,18,18,14로늘지만정확도는대조DeepPsingle2,그외0입니다. 즉이작은gamma고정시간제한실험에서는Deep다중스텝이더안정적으로이어가지만넓은정확도를해결하지는못했습니다. 이것을모든ANN구조나모든gamma에서의보편적우열로일반화하지않습니다.')
    image(s,'14_latest_completion',40,176,880,269)
    foot(s,'정확도 통과(raw): Deep P 단일 1, 나머지 0. 여기의 막대는 완주 수다.')

    s=new('21 · 재현성과 보존','실험 결과와 계산 예산을 함께 남기는 파이프라인을 만들었다',
          '관리형 클라우드 · 순차 학습 · checkpoint + Adam · 단기 게시 gate · 검증된 보고서',
          ['kinetics/ann_multistep_followup.py','results/ann_multistep_long/run_001/complete.json','results/ann_multistep_long/run_001/verification.json'],
          '실험은로컬앱이아니라관리형클라우드에서독립프로세스로진행했습니다. 시간측정에부하를주지않도록모델을순차실행하고active학습시간,update,최종LR,종료이유를기록했습니다. 최신단기와장기는모델및Adam해시를연결하고,단기검증후게시와원격SHA확인gate를통과한뒤장기를시작했습니다. NRMSE는저장된궤적과CSV를재계산해맞는지검사했고완료JSON과파일SHA가연결됩니다. 원격bw의기존결과를보존하면서별도scopedindex로새결과만게시했으며main또는다른브랜치로바꾸지않았습니다. 현재발표는재학습없이저장된원자료에서만도출했습니다.')
    bullets(s,[('비교 가능하게 기록','고정 분할·시간 간격·정규화 · seed · 활성 학습 시간 · 종료 이유'),
               ('결과를 추적 가능하게 보존','모델 / Adam / rollout / NRMSE CSV / 전체·점화 PDF / 검증 SHA'),
               ('실행과 게시를 분리','단기 검증·커밋·원격 SHA 확인 후 장기 시작. 최신 ANN 장기도 게시 완료.')],48,184,854,size=19,gap=24)
    foot(s,'최신 일반 ANN 장기 게시: ad8c537 · 완료 모델 재학습이나 시간 예산 재시작 없이 정리했다.')

    s=new('22 · 다음 실험의 판단 기준','다음 목표는 완주 수보다, 보정 없이 여러 조건에서 맞는 모델이다',
          '아래는 실행한 결과가 아니라, 현재 관찰에 근거한 후속 제안이다.',
          ['results/raw_history_audit/run_001/summary.json','results/ann_multistep_long/run_001/report/test_summary.csv'],
          '앞으로는작은validation범위에서가설을분리해야합니다. 첫째K64에시간4분의1만주는것보다K1/4/16을충분히학습한후늘리는것이유리한지validation-only로확인할수있습니다. 둘째one-stepnormalizedMSE와physicalmulti-variable오차의괴리를줄이는loss가필요합니다. 셋째rollout이train상태를벗어나는지진단하고필요하면그상태에대한Cantera라벨을추가할수있습니다. 다만main에서이미0.4/2ms증강을시도했으므로단순증강을새아이디어라고하지않고다른점과실패원인을명시해야합니다. 마지막으로보존식을출력구조로만족시키는구조를검토하되화학속도정확도를별도로평가해야합니다. 기존test를반복열람했기때문에최종확인은새untouchedholdout이필요합니다. 이페이지의제안은승인없는재학습을시작한다는뜻이아닙니다.')
    table(s,40,179,880,['관찰','다음에 분리해서 검증할 가설'],[
        ['K64에서 최적화 비용이 커짐','validation에서 K / loss 가중치 / 학습 시간 배분을 소규모 비교'],
        ['작은 one-step MSE와 큰 rollout 오차','중간 궤적 오차 + 화학종별 스케일을 함께 점검'],
        ['자기 예측 상태가 학습 분포에서 이탈','분포 이탈 진단 후 필요한 상태에만 정답 보강'],
        ['보정은 보존을 돕지만 kinetics는 틀림','보존 구조와 반응 정확도를 별개로 검증']
    ],ratios=[.4,.6],row_h=53,size=14)
    foot(s,'선택은 validation으로, 최종 확인은 새 holdout으로. 기존 test에서 좋은 사례만 고르지 않는다.',TEAL)

    # Appendix starts after the 23-slide main narrative.
    s=new('Appendix A · 실험 장부','“단기”와 “장기”의 예산이 모든 단계에서 같지는 않다',
          '행별로 독립 연구 범위와 재사용을 구분했다. 신규 모델 총합으로 단순 합산하지 않는다.',
          ['configs/deeponet_long_study.json','configs/element_soft_long.json','configs/weight_long_study.json','configs/gamma_long.json','results/ann_multistep/run_001/config.json'],
          '초기P/T연장,phi연장,soft10연장과3가중치연장은각연구전체최대6시간입니다. gamma장기는모델당150활성초를누적하는별도설정으로전체guard15시간,실제약11시간33분이었으며6시간이라고표현하면안됩니다. gamma단기240개중200개가새학습이고gamma.1의40개는재사용입니다. gamma장기20개는Adam부재때문에기준10k재생성시간도150초안에포함했습니다. 최신시간및ANN비교의short는10k가아니라처음부터150초입니다. 혼합7평가팀은4개학습계열에서파생되며추가7개학습계열이아닙니다.')
    table(s,40,175,880,['연구 단계','단기','장기 / 비교 단위'],[
        ['P-only / T-P / phi','각 구조 10개 · 기준 10k','각 후속 연구 전체 최대6시간'],
        ['Soft λ10','4계열×10 = 40개 · 10k','40개 연장 · 연구 전체 최대6시간'],
        ['Soft λ1/100/1000','3×4계열×10 = 120개 · 10k','120개 연장 · 연구 전체 최대6시간'],
        ['Box–Cox γ 6개','240개 기준점 (신규200/재사용40)','240개 · 추가150활성초/모델'],
        ['시간 DeepONet','3방식×3seed = 9개 · 150초','9개 정확 복원 · 최대150초 추가'],
        ['일반 ANN','2방식×3seed = 6개 · 150초','6개 정확 복원 · 최대150초 추가']
    ],ratios=[.22,.36,.42],row_h=42,size=12.5)
    foot(s,'4개 학습 계열 = ANN / P-only / T-P / T-P-phi. 혼합 평균은 별도 학습 모델을 만들지 않는다.')

    s=new('Appendix B · 수식과 용어','전처리 → 증분 예측 → 물리 상태 복원 → 자기 예측',
          '시간마다 현재 상태를 다시 변환한다. “다음 상태”와 “증분”을 혼동하지 않는다.',
          ['kinetics/soft_constraint_training.py','kinetics/temporal_pilot.py','ANN_DEEPONET_WORKFLOW.md'],
          '기본encode는온도T/1000과질량분율Y의power transform입니다. training current+next에서구한mu,sigma로표준화한좌표를z라할때NN은다음z와현재z의차이를예측합니다. 모델입력에압력등의조건을더합니다. 다음z를inverse해물리상태를얻으며inverse영역이벗어나면raw가실패할수있습니다. multistep은동일map을K번합성해1부터K까지의오차를평균합니다. 실제평가는모든방식에서2us고정입니다. loss의큰값과작은값을서로다른K·gamma·정규화간에절대적인물리정확도로비교하면안됩니다.')
    equations=[('1. 변환','E(T,Y) = [T/1000, Y^γ/γ]'),('2. 표준화','z = (E − μ_train) / σ_train'),('3. 증분 정답','Δz_true = z(t + 2 us) − z(t)'),('4. 다음 상태','z_pred,next = z_current + NN(current, condition)'),('5. 다중 스텝','L_K = mean_{k=1…K} MSE(z_pred,k, z_true,k)')]
    for i,(label,formula) in enumerate(equations):
        yy=178+i*53
        rect(s,40,yy,880,45,PALE,r=4)
        tx(s,label,53,yy+11,185,31,14,TEAL,True)
        tx(s,formula,243,yy+10,660,33,17,bold=True)
    foot(s,'NRMSE는 이 표준화 공간의 loss가 아니라, 복원된 물리 상태와 Cantera를 비교한 지표다.')

    s=new('Appendix C · γ 선택 해석','과거 γ 선택은 보정 포함 온도 기준이며, 전체 화학종 최적은 아니다',
          'validation 완주 수 최대 → 공통 완주 IC의 온도 NRMSE 중앙값 최소 → 동률이면 낮은 γ',
          ['results/gamma_long/run_001/FINAL_ARTIFACTS.json'],
          'gamma스윕당시validation선택규칙은repaired완주수와온도NRMSE였습니다. 선택된gamma는ANN.1,P.2,ANN+P.2,TP.3,ANN+TP.3,phi.2,ANN+phi.2입니다. 이결과를전체반응종의전역최적gamma라고불러서는안됩니다. 이후시간·다중스텝연구는기존P계열을참고해gamma.2를고정했으며새로운gamma최적화는하지않았습니다. 또한lambda는gamma.1에서, gamma는lambda0에서실험해lambda×gamma전체조합은검증하지않았습니다.')
    rows=[['ANN10','.1'],['P-only10 / ANN+P20','.2'],['T/P10 / ANN+T/P20','.3'],['T/P/phi10 / ANN+T/P/phi20','.2']]
    table(s,40,177,489,['평가 설정','선택 γ'],rows,ratios=[.77,.23],row_h=52,size=14)
    bullets(s,[('현재 발표의 정확도 재점검','T + 8반응종을 각각 확인한다.\n온도 선택과 전체 화학종 평가는 다르다.'),('아직 하지 않은 것','λ×γ 전체 factorial 비교,\n일반 ANN 다중 스텝의 γ 최적화.')],559,185,352,size=16,gap=26)
    foot(s,'제시한 γ는 당시 정해둔 규칙의 선택 결과다. 보편적 최적값이나 test 기반 재선택이 아니다.')

    s=new('Appendix D · 변수별 통과 점수','raw 통과 세 설정의 10개 출력 NRMSE를 함께 확인했다',
          '모두 IC39 · 전체 10 ms · 수치는 무차원 비율',
          ['results/raw_history_audit/run_001/passing_variable_NRMSE.csv'],
          '아홉개진단대상출력외에N2까지각각표기했습니다. 세설정의N2도0.1이하이므로N2포함여부가세통과설정을바꾸지않습니다. 그러나질량분율합과원소량최대이탈은여전히존재합니다. 정확도점수가낮아도필요한물리보존기준을별도로설계하고평가해야합니다.')
    image(s,'15_pass_variable_table',40,177,880,205)
    table(s,40,385,880,['같은 세 설정 순서','T/P10 γ=.1','P-only10 γ=.2','P-only3 새 대조'],[
        ['max |ΣY − 1|','0.0129','0.00641','0.0102']
    ],ratios=[.34,.22,.22,.22],row_h=43,size=13)
    foot(s,'원소량 최대 이탈도 약0.0059–0.0116으로 남아 있다. 정확도 통과와 보존식 만족을 구분한다.')

    s=new('Appendix E · 최신 ANN 학습 기록','다중 스텝은 같은 시간에 수행할 수 있는 update 수가 훨씬 적다',
          '장기 종료 시 누적 update · 누적 활성 시간 · 최종 LR · 종료 이유',
          ['results/ann_multistep_long/run_001/training.csv','results/ann_multistep_long/run_001/verification.json'],
          '정확한update와시간은표를따릅니다. multistep의K64누적update수는각seed2929~3563이며single-step보다역전파할시간축이길어한update가비쌉니다. short+long전체활성시간은약1771.3초로29.5분이며평가까지의wall time과구분해야합니다. 모든모델에같은150초추가상한이적용됐지만seed11multi는평탄종료해누적271초입니다. 그외모델은대략300초입니다. 시간상한은평탄도달로표시하지않았습니다.')
    rows=[]
    for r in F['ann_multistep_long_training']:
        rows.append(['단일' if r['method']=='single_step' else '다중',str(r['seed']),f"{r['updates']:,}",f"{r['cumulative_active_seconds']:.1f} s",f"{r['final_learning_rate']:.1e}",'평탄' if r['termination']=='plateau' else '시간상한'])
    table(s,40,175,880,['방식','seed','누적 update','활성 시간','최종 LR','종료'],rows,ratios=[.13,.1,.23,.2,.18,.16],row_h=43,size=14)
    foot(s,'K64 누적 update: 2,929–3,563. 총 활성 학습 약29.5분 (평가·보고서 생성 시간 제외).')

    s=new('Appendix F · 공통 완주 비교','단기와 장기의 오차는 같은 IC 집합에서 비교했다',
          '일반 ANN mean3 · worst9 = T·8반응종 NRMSE 중 최댓값',
          ['results/ann_multistep_long/run_001/report/short_long_common_summary.csv','results/ann_multistep_long/run_001/report/test_short_long_NRMSE.csv'],
          '불완주를제외하는순간완주집합이바뀌어중앙값이좋아보일수있습니다. 그래서동일gamma,동일method,동일mode의공통완주집합에서전후를비교했습니다. rawmulti는IC74하나뿐이어서모집단의확대성과거리가멀고worst9도57이상입니다. repairedmulti의공통12개는전체단기13,장기14완주와다른집합입니다. 일반ANNsingle과multi,DeepONet과ANN을비교할때도중앙값은공통집합기준이어야합니다. raw불완주는full10ms점수없음으로남겼습니다.')
    table(s,40,180,880,['설정','공통 IC 수','단기 worst9','장기 worst9'],[
        ['raw ANN 단일','0','full10ms 비교 없음','full10ms 비교 없음'],
        ['raw ANN 다중','1  (IC74)','62.688','57.656'],
        ['repaired ANN 단일','18','6.431','4.702'],
        ['repaired ANN 다중','12','23.411','23.281']
    ],ratios=[.32,.18,.25,.25],row_h=53,size=14)
    foot(s,'IC가 여러 개면 worst9 중앙값. 모두 0.1 문턱과 비교하면 여전히 크다. 미완주는 0점으로 채우지 않았다.')

    s=new('Appendix G · 결과 그림 찾아보기','원본 그림은 보존하고, 발표용 그림과 검색 목록을 별도로 만들었다',
          '원본 1,283개 PNG/PDF/SVG 파일 + 발표용 16개 그림 묶음',
          ['results/research_story/run_001/all_figures_index.csv','results/research_story/run_001/figures/figure_manifest.json'],
          '원본파일은이동하거나삭제하지않았습니다. 그림과PDF파일1283개는중복보고서버전과여러페이지PDF를포함한파일수이며독립실험수가아닙니다. searchable gallery에서는실험명,main/bw범위,raw/repaired추정분류로필터링할수있습니다. 파일명만으로보정모드를알수없는것은원문확인으로남겼습니다. 발표용16개그림은저장된CSV와궤적에서새로만들었고PNG와벡터PDF를함께줍니다. 원본전체그리드와IC별NRMSE CSV는갤러리의대표자료링크에있습니다. 링크는기준커밋에고정되어후속커밋이생겨도발표근거가바뀌지않습니다.')
    bullets(s,[('발표할 때','이 PPTX / PDF → 한 장에 질문·관찰·다음 이유를 정리'),
               ('질문을 받았을 때','gallery/index.html → 대표 결과·원본 전체 그림 검색'),
               ('수치나 근거를 확인할 때','experiment_registry.csv / all_figures_index.csv / source manifest / IC별 NRMSE')],48,183,852,size=19,gap=24)
    foot(s,'파일 수를 실험 수로 세지 않았다. 발표용 그래프도 저장된 결과만 사용했으며 새 학습·평가는 하지 않았다.')

    s=new('Appendix H · 근거와 커밋','main의 출발점과 bw의 변경 이력을 커밋으로 고정했다',
          '더 자세한 실험 조건·수치·원본 경로는 발표자 노트와 동봉 목록에서 확인할 수 있다.',
          ['results/research_story/run_001/sources/commit_history_complete.json','results/research_story/run_001/sources/used_source_manifest.json'],
          '발표의근거revision은ad8c537입니다. main은e89c656이며,이경계후의bw실험을설명합니다. 각페이지의source와노트는근거파일을GitHub커밋URL에연결하고,읽은원문은GitblobSHA또는SHA256으로확인했습니다. 발표자료자체의게시커밋이추가되어도근거revision은바뀌지않습니다. 사용자요청순서대로일반ANN장기결과를먼저게시한후이자료를작성했습니다. 단기대장과원본전체그림목록,재생성스크립트를함께제공하므로나중에필요한페이지를수정하거나원본을확인할수있습니다.')
    entries=[('main 기준','e89c656','e89c656eeeac0e93bafb4322b9482c7dfec70168'),
             ('초기 DeepONet','3affe75','3affe7545a3946075a7f80547d70cb6147c15c06'),
             ('γ 장기 결과','7310311','7310311489da157c9e4f869c1f9ee4c62dc3a017'),
             ('시간 DeepONet 장기','53df7a5','53df7a5a54b1373110269a98a8d6b85e93763aac'),
             ('일반 ANN 단기','dade4e6','dade4e601585ebfc6866aef92995680cbe669368'),
             ('일반 ANN 장기 · 최신','ad8c537',REV)]
    for i,(label,short,full) in enumerate(entries):
        yy=179+i*43
        rect(s,40,yy,880,36,PALE if i%2==0 else WHITE)
        tx(s,label,53,yy+9,379,25,14,bold=True)
        tx(s,short+'  · GitHub commit',451,yy+9,450,25,14,BLUE,link=BASE+'/commit/'+full)
    foot(s,'자료의 결론: 특정 한 조건의 성공을 전체 조건의 성공으로 확대하지 않고, 다음 실험의 가설을 좁혔다.',TEAL)


def runs(text,bold=False):
    result=[]
    for ch in text:
        font=font_for(ch,bold)
        if result and result[-1][0]==font:result[-1]=(font,result[-1][1]+ch)
        else:result.append((font,ch))
    return result


def render():
    overrides=json.loads((ROOT/'speaker_notes.json').read_text())
    for number,text in overrides.items():SLIDES[int(number)-1]['notes']=text
    prs=Presentation();prs.slide_width=Pt(W);prs.slide_height=Pt(H)
    prs.core_properties.title='수소 반응계 NN 연구: main에서 bw까지'
    prs.core_properties.subject='실험의 이유, 방법, raw/repaired 결과, 남은 문제'
    prs.core_properties.author='rom-cfd research'
    pdf=canvas.Canvas(str(OUT/'hydrogen_research_story.pdf'),pagesize=(W,H),pageCompression=1)
    pdf.setTitle(prs.core_properties.title);pdf.setAuthor('rom-cfd research')
    notes=[]; source_map=[]
    for number,s in enumerate(SLIDES,1):
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        footer=f'main → bw  |  evidence {REV[:7]}  |  {number:02d} / {len(SLIDES):02d}'
        tx(s,footer,40,516,880,16,size=8.5,color=GRAY)
        for e in s['elements']:
            x,y,w,h=e['x'],e['y'],e['w'],e['h']
            if e['kind']=='rect':
                shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if e['r'] else MSO_SHAPE.RECTANGLE,Pt(x),Pt(y),Pt(w),Pt(h))
                shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(e['fill'].lstrip('#'))
                if e['line']:shape.line.color.rgb=RGBColor.from_string(e['line'].lstrip('#'))
                else:shape.line.fill.background()
                pdf.setFillColor(HexColor(e['fill']));pdf.setStrokeColor(HexColor(e['line'] or e['fill']))
                pdf.roundRect(x,H-y-h,w,h,e['r'],fill=1,stroke=bool(e['line']))
            elif e['kind']=='image':
                slide.shapes.add_picture(e['path'],Pt(x),Pt(y),width=Pt(w),height=Pt(h))
                pdf.drawImage(ImageReader(e['path']),x,H-y-h,w,h,mask='auto')
            else:
                shape=slide.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(h))
                tf=shape.text_frame;tf.clear();tf.word_wrap=False;tf.margin_left=0;tf.margin_right=0;tf.margin_top=0;tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.TOP
                for i,line in enumerate(e['lines']):
                    p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
                    p.line_spacing=Pt(e['size']*e['leading']);p.space_before=Pt(0);p.space_after=Pt(0)
                    xx=x; baseline=H-y-e['size']*.92-i*e['size']*e['leading']
                    for font,text in runs(line,e['bold']):
                        run=p.add_run();run.text=text;run.font.name=fonts[font][1];run.font.size=Pt(e['size']);run.font.bold=e['bold'];run.font.color.rgb=RGBColor.from_string(e['color'].lstrip('#'))
                        if e['link']:run.hyperlink.address=e['link']
                        pdf.setFont(font,e['size']);pdf.setFillColor(HexColor(e['color']));pdf.drawString(xx,baseline,text);xx+=pdfmetrics.stringWidth(text,font,e['size'])
                if e['link']:pdf.linkURL(e['link'],(x,H-y-h,x+w,H-y),relative=0,thickness=0)
        source_links=[]
        for path in s['sources']:
            # The presentation's own new paths resolve within the distributed package.
            link=src(path) if not path.startswith('results/research_story/') else path.replace('results/research_story/run_001/','../')
            source_links.append(f'{path}\n  {link}')
            source_map.append(dict(slide=number,title=s['title'].replace('\n',' '),path=path,url=link))
        fullnote=f"{number:02d}. {s['title']}\n\n{s['notes']}\n\n근거\n"+'\n'.join(source_links)
        slide.notes_slide.notes_text_frame.text=fullnote
        notes.append(fullnote)
        pdf.bookmarkPage(f'slide{number}');pdf.addOutlineEntry(f'{number:02d} {s["title"].replace(chr(10)," ")}',f'slide{number}',level=0)
        pdf.showPage()
    prs.save(OUT/'hydrogen_research_story.pptx');pdf.save()
    (ROOT/'발표자_설명노트.txt').write_text('\n\n'+'\n\n'+'\n\n'.join(notes),encoding='utf-8')
    (ROOT/'sources/slide_content.json').write_text(json.dumps(SLIDES,ensure_ascii=False,indent=2))
    (ROOT/'sources/layout_checks.json').write_text(json.dumps(FIT,ensure_ascii=False,indent=2))
    with (ROOT/'slide_sources.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['slide','title','path','url'],lineterminator='\n');writer.writeheader();writer.writerows(source_map)
    print(f'Created {len(SLIDES)} slides. All {len(FIT)} text bounds passed.')


if __name__=='__main__':
    build();render()
