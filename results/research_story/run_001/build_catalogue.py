"""Build the offline, searchable research gallery and scoped experiment ledger."""
import csv
import html
import json
from pathlib import Path
import pandas as pd
from prepare_facts import ROOT,REV,MAIN,TREE,REFS,source,nrmse_stats

BASE=f'https://github.com/rom-cfd/personal/blob/{REV}/'
F=json.loads((ROOT/'sources/facts.json').read_text())


def write_csv(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    registry=[]
    def add(stage,scope,question,change,models,budget,result,root,why_next):
        registry.append(dict(stage=stage,scope=scope,question=question,change=change,trained_model_scope=models,
            budget=budget,observed_result=result,why_next=why_next,root=root,evidence_url=BASE+root))
    add('00 main','main 기준','10개 대신 100개를 평균하면 개선되는가','100-member ANN과 앞10개 부분집합 비교',
        '기존100개 재사용; 평균/medoid는 평가 조합','main의 기존 예산; bw 신규 학습으로 세지 않음',
        '물리 보정 포함 mean10/100 모두18완주; T RMSE179.1/275.2K; raw100 결과 없음',
        'results/committee100_long10ms','앙상블 확대만으로 해결되지 않아 구조 비교')
    add('01 조건부 구조','bw','압력·온도를 분리하면 개선되는가','ANN10 대 T/P Deep10, 혼합20',
        'T/P10 신규; ANN10은 기존100의 앞10','기준10,000 update',
        '좋은 온도 사례가 있었으나 전체 test의 화학종 정확도는 별도 점검 필요',
        'results/ann_deeponet_mean10','저장되지 않은 P-only 모델 재생성 후 구조·장기 비교')
    add('02 P-only 재생성','bw','P-only 모델 자체를 다시 만들 수 있는가','P-only Deep10 기준점 생성',
        'P-only10 신규','10,000 update / 사용자 승인 전체최대10분',
        '기준 checkpoint와 기록 확보; raw 기준은 후속 감사에 포함',
        'results/p_only_rebuild_training_checks','P-only/T-P에 장기 학습을 적용해 최적화 부족 점검')
    add('03 P/T 장기','bw','loss 감소 부족이 원인인가','P-only10와 T/P10 순차 연장',
        '20개 연장; ANN 기준 재사용','연구 전체최대6시간; 평탄도달0',
        'raw 완주 ANN/P/ANN+P/TP/ANN+TP=2/6/9/12/6; 정확도는TP IC39만1',
        'results/deeponet_long_study/run_001','초기 phi 조건을 추가해 상태 표현 점검')
    add('04 T/P/phi','bw','초기 혼합비 정보가 도움이 되는가','trunk에 초기phi 추가; short/long/혼합',
        'phi10 신규 +10연장; 나머지 기준 재사용','단기10k; 장기 연구 전체최대6시간',
        'raw phi 단기6→장기7완주, 혼합12→11; 정확도통과0',
        'results/tp_phi_study/run_001','물리 정보를 손실에 넣어 이탈 억제 시도')
    add('05 λ10','bw','원소 보존 손실이 rollout에 도움이 되는가','MSE + 다음 상태 원소량L1, gamma=.1',
        '4계열×10=40신규 +40연장; 혼합별도학습없음','단기10k; 장기 연구 전체최대6시간',
        '새 soft 모델 raw 정확도통과0; 장기평탄0/시간상한40',
        'results/element_soft_long/run_001','가중치가 원인인지 λ1/100/1000 비교')
    add('06 λ1/100/1000','bw','물리 벌점 가중치를 바꾸면 개선되는가','gamma=.1 고정; 3개λ short/long',
        '3×4계열×10=120신규 +120연장','단기10k; 장기 연구 전체최대6시간',
        '새 soft 모델 raw 정확도통과0; 장기평탄0/시간상한120',
        'results/weight_long_study/run_001','soft 항을 빼고 전처리 gamma만 분리해 비교')
    add('07 γ 단기','bw','미량종 스케일을 바꾸면 개선되는가','gamma=.05/.1/.2/.3/.5/1, λ0',
        '240기준점: 신규200 +gamma.1의40재사용','10,000 update 기준',
        'raw 정확도통과0; gamma에 따른 완주 편차',
        'results/gamma_sweep/run_001','최적화 부족과 변환 효과를 나눠 보기 위해 장기 연장')
    add('08 γ 장기','bw','같은 추가 계산 시간에서도 γ 차이가 남는가','240기준점 순차 연장',
        '220정확 Adam복원; gamma.1의20기준10k재생성비용도 예산포함','추가활성150초/모델; 전체guard15시간; 실제약11시간33분',
        'raw P-only gamma.2 IC39만1통과; repaired 최대3/18; 평탄0/시간상한240',
        'results/gamma_long/run_001','온도/보정 성과와 raw 화학종 정확도를 분리해서 감사')
    add('09 raw 감사','bw','보정 없이 맞은 조건이 실제 얼마나 있는가','저장된 held-out raw 결과를 RMS NRMSE로 재검증',
        '새 모델 학습 없음; 반복baseline을 독립실험으로 합산하지 않음','저장된 CSV/trace만 사용',
        '3설정×IC 통과 모두IC39; 고유1/18; 초기0.1ms까지 통과2설정',
        'results/raw_history_audit/run_001','훈련과 자기 예측의 차이를 줄이는 시간 학습 시도')
    add('10 시간 DeepONet','bw','시간 간격 조건과 다중 스텝 loss가 도움이 되는가','single/interval/multi, seed11/22/33, gamma=.2',
        '9신규 +9연장; mean3는 평가 조합','단기150활성초 +추가최대150초/모델; multi장기K64',
        'raw mean3 완주 single7→9,interval0→0,multi14→13; 정확도0→1/0/0',
        'results/temporal_pilot_long/run_001','다중 스텝 효과를 일반 ANN에서 별도 검증')
    add('11 일반 ANN','bw','같은 다중 스텝 loss가 일반 MLP에서도 도움이 되는가','11-64-64-64-10 tanh; single/multi; 같은seed 초기값',
        '6신규 +6정확 Adam연장; mean3는 평가 조합','단기150활성초 +추가최대150초/모델; 장기평탄1/상한5',
        'raw mean3 single0→0,multi1→2완주; 정확도는 seed/mean3 모두0',
        'results/ann_multistep_long/run_001','새 정확도 조건을 추가하지 못함; validation에서 다음 가설 분리 필요')
    write_csv('experiment_registry.csv',registry)
    write_csv('commit_history.csv',[dict(date_KST=c['date_KST'],sha=c['sha'],title=c['message'].splitlines()[0],url='https://github.com/rom-cfd/personal/commit/'+c['sha'],scope='main 기준' if c['sha']==MAIN else 'bw 변경') for c in F['history']])

    overview=[]
    for label in ['lambda_short','gamma','soft10']:
        for r in F[label]:
            penalty=0 if label=='gamma' else r.get('weight',10 if r.get('training','')!='baseline' else 0)
            overview.append(dict(study=label,stage=r.get('stage',r.get('training','10k')),setting=r['team'],gamma=r.get('gamma',.1),lambda_value=penalty,
                mode=r['mode'],complete=r['complete'],accuracy_passes=r['passed'],total=18,predictor_scope='10-member or mixed20; see registry'))
    for family,rows in F['series'].items():
        for r in rows:
            overview.append(dict(study=family,stage='long' if family.endswith('long') else 'short150s',setting=r['method'],gamma=.2,lambda_value=0,mode=r['mode'],complete=r['complete'],accuracy_passes=r['joint9_le_0p1'],total=18,predictor_scope='mean3'))
    # Independently read the three long-weight reports; avoid counting repeated λ0 baselines.
    weight_rows=[]
    root=F['weight_long_study_artifacts']['report']
    for weight in [1,100,1000]:
        for mode in ['raw','repaired']:
            path=f'{root}/lambda_{weight}/{mode}/per_IC_NRMSE.csv'
            d=nrmse_stats(path)
            for (training,team),g in d.groupby(['stage','team']):
                row=dict(study='weight_long',stage=training,setting=team,gamma=.1,lambda_value=0 if training=='baseline' else weight,mode=mode,complete=int(g.complete.sum()),accuracy_passes=int(g.passed.sum()),total=18,predictor_scope=f'10-member or mixed20; baseline rows repeat in lambda{weight} report')
                overview.append(row);weight_rows.append(row)
    assert all(r['accuracy_passes']==0 for r in weight_rows if r['mode']=='raw' and r['stage']!='baseline')
    write_csv('evaluation_overview.csv',overview)

    figures=json.loads((ROOT/'figures/figure_manifest.json').read_text())
    inventory=json.loads((ROOT/'sources/figure_inventory.json').read_text())
    cards=[]
    for i,v in enumerate(figures,1):
        caption=v['caption']
        for a,b in [('약0.0068K','약 0.0068 K'),('약2.02K','약 2.02 K'),('모두IC39','모두 IC39'),('모두0','모두 0'),('세설정','세 설정'),('세설정은','세 설정은'),('평탄기준','평탄 기준'),('정확한10ms','정확한 10 ms'),('짧은예측부터길게확장한다','짧은 예측부터 길게 확장한다'),('장기는최대150초추가이고평탄종료면더짧다','장기는 최대 150초 추가이며 평탄 종료면 더 짧다')]:caption=caption.replace(a,b)
        sources=[]
        for path in v['sources']:
            if path in TREE:link=BASE+path
            elif (ROOT/'sources'/path).exists():link='../sources/'+path
            else:link='../sources/selected_trace_provenance.json'
            sources.append(f'<a href="{html.escape(link)}">{html.escape(Path(path).name)}</a>')
        cards.append(f'''<article class="figure" data-group="{html.escape(v['group'])}"><div class="eyebrow">{i:02d} · {html.escape(v['group'])}</div><a href="../figures/{v['name']}.png"><img loading="lazy" src="../figures/{v['name']}.png" alt="{html.escape(caption)}"></a><h3>{html.escape(caption)}</h3><p class="links"><a href="../figures/{v['name']}.png">PNG 보기</a><a href="../figures/{v['name']}.pdf">벡터 PDF</a></p><p class="source">근거: {' · '.join(sources)}</p></article>''')
    quick=[]
    for family,label in [('ann_multistep_long','최신 일반 ANN 단기·장기'),('temporal_pilot_long','시간 DeepONet 단기·장기')]:
        root=f'results/{family}/run_001/report/'
        names=[('보정 없는 전체18케이스 PDF','profiles_test_raw.pdf'),('보정 없는 점화확대 PDF','profiles_test_raw_ignition_zoom.pdf'),('보정 포함 전체18케이스 PDF','profiles_test_repaired.pdf'),('보정 포함 점화확대 PDF','profiles_test_repaired_ignition_zoom.pdf'),('IC별 NRMSE','NRMSE_by_IC.csv'),('공통완주 전후 비교','short_long_common_summary.csv'),('완주·정확도 요약','test_summary.csv'),('완주 범위 그림','coverage.png'),('고정 IC39 raw 확대','diagnostic_IC39_raw_zoom.png')]
        if family=='ann_multistep_long':names.append(('Loss·LR 그림','loss_and_lr.pdf'))
        assert all(root+n in TREE for _,n in names)
        quick.append(f'<div class="link-card"><h3>{label}</h3>'+''.join(f'<a href="{BASE+root+name}">{text}</a>' for text,name in names)+'</div>')
    for key,label in [('gamma_long','γ 장기 전체 보고서'),('element_soft_long','λ10 장기 보고서'),('weight_long_study','λ1/100/1000 장기 보고서')]:
        root=F[key+'_artifacts']['report']
        names=[p for p in TREE if p.startswith(root+'/') and p.endswith('REPORT_KO.md')]
        quick.append(f'<div class="link-card"><h3>{label}</h3><a href="{BASE+(names[0] if names else root)}">보고서와 원본 자료 열기</a></div>')
    groups=sorted({f['group'] for f in figures})
    html_text='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>수소 반응계 main → bw 연구 자료</title><style>
@font-face{font-family:Nanum;src:url('../fonts/NanumGothic-Regular.ttf')}@font-face{font-family:Nanum;src:url('../fonts/NanumGothic-Bold.ttf');font-weight:700}
*{box-sizing:border-box}body{font-family:Nanum,Arial,sans-serif;margin:0;background:#f4f7fb;color:#10243a;line-height:1.7}a{color:#2563eb;text-underline-offset:3px}header{padding:44px max(24px,calc((100vw - 1180px)/2));background:#10243a;color:white}header p{color:#c4d5e6;max-width:920px}h1{font-size:clamp(28px,4vw,43px);line-height:1.35;margin:12px 0}h2{font-size:25px;margin:36px 0 12px}h3{font-size:16px;line-height:1.65}nav{display:flex;gap:20px;flex-wrap:wrap}nav a{color:#b9d9ff}main{max-width:1228px;padding:12px 24px 60px;margin:auto}.eyebrow{font-size:12px;letter-spacing:.05em;color:#087f8c;font-weight:700}.downloads,.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.download,.stat{background:white;padding:22px;border:1px solid #d6e0ea;border-radius:10px}.download a{font-size:19px;font-weight:700}.stat strong{display:block;font-size:30px;color:#087f8c}.note{background:#e9eff7;border-left:4px solid #087f8c;padding:16px 20px}.figures{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}.figure{padding:18px;background:white;border:1px solid #d6e0ea;border-radius:10px}.figure img{width:100%;height:265px;object-fit:contain;background:white}.links{display:flex;gap:18px}.source{font-size:11px;color:#63758a;overflow-wrap:anywhere}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}input,select,button{font:inherit;padding:10px 12px;border:1px solid #b9c7d6;border-radius:6px;background:white}input{min-width:260px;flex:1}button{cursor:pointer}button:disabled{opacity:.4}table{border-collapse:collapse;width:100%;background:white;font-size:12px}th{background:#10243a;color:white;text-align:left}th,td{padding:12px;vertical-align:top;border-bottom:1px solid #e5ebf3}td.path{max-width:620px;overflow-wrap:anywhere}.table-scroll{overflow:auto}.pager{display:flex;gap:16px;align-items:center;margin:14px 0}.quick{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.link-card{padding:18px;background:white;border:1px solid #d6e0ea;border-radius:8px}.link-card a{display:block;font-size:13px;margin:5px 0}.small{font-size:12px;color:#63758a}footer{margin-top:40px;color:#63758a;font-size:12px}details{padding:14px;background:white;border:1px solid #d6e0ea}summary{cursor:pointer;font-weight:700}@media(max-width:760px){.figures,.quick,.downloads,.stats{grid-template-columns:1fr}.figure img{height:auto}header{padding:30px 24px}input{min-width:160px}}
</style></head><body><header><div class="eyebrow" style="color:#85ccd3">HYDROGEN KINETICS · RESEARCH STORY</div><h1>main에서 bw까지,<br>실험의 이유와 결과를 한 흐름으로</h1><p>조건부 모델·물리 벌점·전처리·시간 학습·일반 ANN을 비교한 기록입니다. 완주와 정확도, raw와 물리 보정을 나눠 봅니다.</p><nav><a href="#start">발표 자료</a><a href="#curated">발표용 그림</a><a href="#reports">전체 결과</a><a href="#archive">원본 그림 검색</a></nav></header><main>
<section id="start"><div class="downloads"><div class="download"><a href="../slides/hydrogen_research_story.pdf">발표 PDF</a><p>31장 · 본문23 + 부록8</p></div><div class="download"><a href="../slides/hydrogen_research_story.pptx">편집용 PowerPoint</a><p>편집 가능한 글·도형, 발표자 노트 포함</p></div><div class="download"><a href="../발표자_설명노트.txt">발표 설명 노트</a><p>각 실험의 이유·관찰·다음 질문</p></div></div><div class="stats"><div class="stat"><strong>1 / 18</strong>보정 없이 정확도 통과한 고유 IC<br><span class="small">세 설정 모두 IC39</span></div><div class="stat"><strong>16</strong>발표용 PNG + 벡터 PDF 묶음</div><div class="stat"><strong>1,283</strong>색인된 원본 PNG/PDF/SVG 파일<br><span class="small">독립 실험 수가 아님</span></div></div><p class="note">NRMSE = RMSE(Cantera, NN) / RMS(Cantera). 백분율이 아닌 무차원 비율입니다. 정확도 통과는 full10ms + T·8반응종 모두 ≤0.1인 사후 진단이며, 불완주에 전체 점수를 주지 않습니다. repaired는 clipping·원소 투영·엔탈피 온도 보정·유효 후보 선택을 포함한 별도 시스템입니다.</p><p><a href="../experiment_registry.csv">실험 장부 CSV</a> · <a href="../evaluation_overview.csv">평가 수치 종합 CSV</a> · <a href="../slide_sources.csv">슬라이드별 근거</a> · <a href="../sources/commit_history_complete.json">커밋 이력</a></p></section>
<section id="curated"><h2>발표용으로 정리한 그림</h2><p>저장된 CSV와 궤적에서 다시 그렸습니다. 원본 결과를 이동·삭제하거나 새 학습을 실행하지 않았습니다.</p><div class="controls"><select id="curatedFilter"><option value="">모든 주제</option>__GROUPS__</select></div><div class="figures">__CARDS__</div></section>
<section id="reports"><h2>전체 결과 · IC별 수치</h2><p>아래 링크는 근거 커밋에 고정된 GitHub 원본입니다. 전체 PDF는 18개 IC를 각각 확인하도록 구성됩니다.</p><div class="quick">__QUICK__</div></section>
<section id="archive"><h2>원본 그림·PDF 1,283개 검색</h2><p>원본은 GitHub에서 엽니다. 버전별 보고서, 여러 페이지 PDF와 중복 기준 그림이 포함되어 파일 수를 학습 모델 수나 독립 실험 수로 더하면 안 됩니다.</p><div class="controls"><input id="search" aria-label="파일 경로 검색" placeholder="예: gamma_0.2, IC39, coverage, raw"><select id="experiment"><option value="">전체 실험</option></select><select id="scope"><option value="">main / bw 모두</option><option>main 기준 자료</option><option>bw 추가/변경 자료</option></select><select id="mode"><option value="">모든 보정 구분</option><option>raw</option><option>repaired</option><option>원문 확인</option></select></div><p class="small">raw/repaired 분류는 경로에서 추정했습니다. ‘원문 확인’은 미보정이라는 뜻이 아닙니다. main/bw 구분은 main 파일 blob과 비교했습니다.</p><div class="pager"><button id="prev">이전</button><span id="count"></span><button id="next">다음</button><a href="../all_figures_index.csv">전체 목록 CSV</a></div><div class="table-scroll"><table><thead><tr><th>실험</th><th>범위 / 모드</th><th>원본 경로</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<footer>근거 revision: __REV__ · main 경계: __MAIN__<br>최신 일반 ANN 장기 게시: <a href="https://github.com/rom-cfd/personal/commit/__REV__">__REV7__</a><br>ZIP에는 발표·선별 그림·근거 자료와 전체 색인을 담았습니다. 원본 전체 대용량 checkpoint/그림은 저장소의 고정 링크를 사용합니다.</footer></main>
<script type="application/json" id="inventory">__DATA__</script><script>
const data=JSON.parse(document.getElementById('inventory').textContent);const $=id=>document.getElementById(id);let page=0;const size=40;
for(const name of [...new Set(data.map(x=>x.experiment))].sort()){const o=document.createElement('option');o.textContent=name;$('experiment').append(o)}
function selected(){const q=$('search').value.toLowerCase();return data.filter(x=>(!q||x.path.toLowerCase().includes(q))&&(!$('experiment').value||x.experiment===$('experiment').value)&&(!$('scope').value||x.scope===$('scope').value)&&(!$('mode').value||x.mode===$('mode').value))}
function render(){const a=selected();page=Math.min(page,Math.max(0,Math.ceil(a.length/size)-1));$('rows').replaceChildren();for(const x of a.slice(page*size,(page+1)*size)){const tr=document.createElement('tr');for(const value of [x.experiment,x.scope+' / '+x.mode]){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td');td.className='path';const link=document.createElement('a');link.href=x.url;link.textContent=x.path;td.append(link);tr.append(td);$('rows').append(tr)}$('count').textContent=a.length?`${a.length}개 중 ${page*size+1}–${Math.min((page+1)*size,a.length)}`:'검색 결과 없음';$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=a.length}
for(const id of ['search','experiment','scope','mode'])$(id).addEventListener('input',()=>{page=0;render()});$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};$('curatedFilter').onchange=()=>{for(const el of document.querySelectorAll('.figure'))el.hidden=$('curatedFilter').value&&el.dataset.group!==$('curatedFilter').value};render();
</script></body></html>'''
    for token,value in {'__GROUPS__':''.join(f'<option>{html.escape(g)}</option>' for g in groups),'__CARDS__':''.join(cards),'__QUICK__':''.join(quick),'__REV__':REV,'__MAIN__':MAIN,'__REV7__':REV[:7],'__DATA__':json.dumps(inventory,ensure_ascii=False).replace('</',r'<\/')}.items():html_text=html_text.replace(token,value)
    (ROOT/'gallery/index.html').write_text(html_text,encoding='utf-8')
    readme=f'''수소 반응계 연구 발표 자료 — main → bw
기준 날짜: 2026-11-06 (KST)
근거 bw revision: {REV}
main 경계: {MAIN}

먼저 볼 파일
1. slides/hydrogen_research_story.pdf — 발표용 31장 (본문23 + 부록8)
2. slides/hydrogen_research_story.pptx — 글/도형 편집 가능, 발표자 노트 포함
3. gallery/index.html — 브라우저에서 열면 발표용16그림과 원본1,283파일을 검색
4. 발표자_설명노트.txt — 실험 이유 → 변경 → 관찰 → 다음 단계 설명
5. experiment_registry.csv — 연구 단계별 예산, 재사용, 결과, 근거
6. evaluation_overview.csv — raw/repaired 완주·정확도 수치
7. all_figures_index.csv — 원본 그림/PDF/SVG의 고정 GitHub 경로
8. slide_sources.csv — 페이지별 근거 경로
9. sources/evidence.zip — 원문71개를 변경 없이 보관 (내부 경로=원래 저장소 경로)

폰트와 편집
PPTX는 NanumGothic과 수식 기호용 DejaVu Sans를 사용합니다. fonts/의 TTF를 설치하면 동일하게 표시됩니다.
PDF에는 폰트를 포함했으므로 설치하지 않아도 됩니다. 그림은 PNG와 벡터 PDF를 제공합니다.
전체 원본 checkpoint와 그림 수 GB를 ZIP에 복제하지 않았습니다. 원본은 고정 GitHub 링크에서 엽니다.
갤러리의 검색과 발표용 그림은 오프라인에서 작동하며 원본 링크는 네트워크가 필요합니다.

해석 원칙
- main의 데이터/100-member ANN을 bw 신규 성과로 합산하지 않았습니다.
- NRMSE = RMSE(Cantera, NN)/RMS(Cantera), %가 아닌 무차원 비율입니다.
- 정확도 통과는 full10ms + T·8반응종 각각NRMSE≤.1인 사후 진단입니다. N2는 별도입니다.
- 불완주에는 full10ms 점수가 없습니다. 완주, 정확도, 보존 오차는 별개입니다.
- raw는 보정 없는 후보, repaired는 clipping/원소투영/엔탈피온도복원/유효후보평균을 포함합니다.
- soft constraint는 학습 중 원소량 L1 벌점입니다. repaired나 반응률 직접 회귀와 다릅니다.
- 과거 γ 선택은 repaired validation의 완주·온도 기준입니다. 화학종 전체 최적이라고 주장하지 않습니다.
- 최신 temporal/ANN의 단기는150활성초이며, 초기 연구의10k 단기와 예산이 다릅니다.
- 반복 baseline, 혼합 평가 조합과 재사용 모델을 새 학습 모델로 더하지 않았습니다.
- 원본1,283파일은 독립 실험 수가 아니며 unrelated handoff_audit_20261102는 제외했습니다.
- 제안한 다음 실험은 미실행입니다. 저장된 결과만 사용했으며 새 학습/평가는 하지 않았습니다.

자료의 재생성
원래 저장소/환경에서 collect_sources.py → prepare_facts.py → make_figures.py →
build_presentation.py → build_catalogue.py → verify_package.py 순서입니다.
prepare_facts.py는 원본 trace의 blob SHA를 대조합니다. 원본 데이터·모델은 수정하지 않습니다.
build_presentation.py만 실행하면 동봉된 facts/그림에서 PPTX/PDF를 다시 만들 수 있습니다.
PDF와 PPTX는 같은 배치 사양을 사용해 생성했고 PDF31페이지를 렌더링해 확인했습니다.
'''
    (ROOT/'읽어주세요.txt').write_text(readme,encoding='utf-8')
    used={v['path']:v for v in json.loads((ROOT/'sources/used_source_manifest.json').read_text())}
    used.update(REFS)
    (ROOT/'sources/used_source_manifest.json').write_text(json.dumps(list(used.values()),indent=2))
    print('Catalogue built:',len(registry),'ledger stages;',len(overview),'evaluation rows;',len(inventory),'original files')


if __name__=='__main__':main()
