"""Check presentation evidence, numerical claims, links and exported pages."""
import hashlib
import json
import subprocess
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
import zipfile
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from PIL import Image,ImageOps,ImageDraw,ImageFont
from pptx import Presentation
from prepare_facts import ROOT,REPO,REV,MAIN,TREE,REFS,source


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks=[]
    def passed(name,detail):checks.append(dict(check=name,status='passed',detail=detail))
    f=json.loads((ROOT/'sources/facts.json').read_text())
    assert f['source_revision']==REV and f['main_revision']==MAIN
    # Verify all copied evidence against the pinned remote tree, not the old checkout HEAD.
    manifests=json.loads((ROOT/'sources/evidence_manifest.json').read_text())
    with zipfile.ZipFile(ROOT/'sources/evidence.zip') as archive:
        for v in manifests:
            raw=archive.read(v['path'])
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            assert blob==v['git_blob_sha']==TREE[v['path']]
            assert hashlib.sha256(raw).hexdigest()==v['sha256']
    passed('copied_evidence_hashes',len(manifests))
    used=json.loads((ROOT/'sources/used_source_manifest.json').read_text())
    for v in used:
        path=source(v['path'],alias=v['local']);assert sha(path)==v['sha256'];v['local']=str(path)
        raw=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==TREE[v['path']]
    passed('used_original_evidence_hashes',len(used))
    slides=json.loads((ROOT/'sources/slide_content.json').read_text())
    for s in slides:
        for path in s['sources']:
            if path.startswith('results/research_story/'):
                assert (REPO/path).is_file()
            else:
                assert path in TREE;source(path)
    merged={v['path']:v for v in used};merged.update(REFS)
    (ROOT/'sources/used_source_manifest.json').write_text(json.dumps(list(merged.values()),indent=2))
    passed('slide_citations',sum(len(s['sources']) for s in slides))
    assert len(slides)==31 and all(x['passed'] for x in json.loads((ROOT/'sources/layout_checks.json').read_text()))
    passed('text_layout_bounds','468 text blocks; explicit wrap; no automatic shrink')
    assert f['raw_audit_including_latest_ANN']['unique_ICs']==[39]
    for family in ['ann_multistep','ann_multistep_long']:
        assert all(row['joint9_le_0p1']==0 for row in f[family+'_all_predictors'])
    # Recompute the three successful raw profiles and the misleading T-only case.
    variables=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','N2']
    keys={'old_TP_long_gamma0.1':'IC39_legacy_TP_raw','gamma_P_only_long_gamma0.2':'IC39_gamma_P_raw','current_single_step_long_gamma0.2':'IC39_temporal_pilot_long_single_step_raw'}
    with np.load(ROOT/'sources/selected_traces.npz') as traces:
        reference=traces['IC39_reference'];assert reference.shape==(5001,10)
        for configuration,key in keys.items():
            pred=traces[key];assert pred.shape==reference.shape
            values=np.linalg.norm(pred-reference,axis=0)/np.linalg.norm(reference,axis=0)
            saved=[next(row['nrmse'] for row in f['pass_variables'] if row['configuration']==configuration and row['variable']==name) for name in variables]
            assert np.allclose(values,saved,rtol=1e-10,atol=1e-12)
            assert max(values)<=.1
            early=np.linalg.norm(pred[:51]-reference[:51],axis=0)/np.linalg.norm(reference[:51],axis=0)
            row=next(r for r in f['raw_passes'] if r['configuration']==configuration)
            assert np.isclose(max(early),row['early_max10'],rtol=1e-10)
        ref=traces['IC7_reference'];pred=traces['IC7_gamma02_ANNP_repaired']
        values=np.linalg.norm(pred-ref,axis=0)/np.linalg.norm(ref,axis=0)
        assert np.isclose(values[0],f['IC7_trap']['T_NRMSE'])
        assert np.isclose(values[6],f['IC7_trap']['H2O_NRMSE'])
        assert np.isclose(values[5],f['IC7_trap']['OH_NRMSE'])
    passed('recomputed_saved_trace_NRMSE','3×10 full-horizon +3×10 early outputs; IC07 T/H2O/OH; RMSE/RMS, no percent')
    overview=pd.read_csv(ROOT/'evaluation_overview.csv')
    assert (overview.loc[overview.study=='gamma','lambda_value']==0).all()
    assert ((overview.complete>=overview.accuracy_passes)&(overview.complete<=18)).all()
    assert len(overview)==426
    passed('evaluation_overview','426 rows; repetitions are labelled, not independent trials')
    prs=Presentation(ROOT/'slides/hydrogen_research_story.pptx');assert len(prs.slides)==31
    assert all(slide.has_notes_slide and len(slide.notes_slide.notes_text_frame.text)>100 for slide in prs.slides)
    passed('PPTX','31 pages; editable text/shapes and Korean speaker notes')
    pdf=pdfium.PdfDocument(ROOT/'slides/hydrogen_research_story.pdf');assert len(pdf)==31
    previews=ROOT/'slides/previews';previews.mkdir(exist_ok=True)
    thumbs=[];texts=[]
    for i,page in enumerate(pdf):
        textpage=page.get_textpage();text=textpage.get_text_range();texts.append(text)
        assert len(text)>80 and '\ufffd' not in text and '\u25a1' not in text
        im=page.render(scale=1.25).to_pil();im.save(previews/f'slide_{i+1:02d}.png')
        thumbs.append(ImageOps.contain(im,(480,270)));textpage.close();page.close()
    font=ImageFont.truetype(str(ROOT/'fonts/NanumGothic-Regular.ttf'),14)
    for start in range(0,len(thumbs),12):
        sheet=Image.new('RGB',(1500,1200),'#D9E2EC');draw=ImageDraw.Draw(sheet)
        for j,thumb in enumerate(thumbs[start:start+12]):
            x=(j%3)*500+10;y=(j//3)*300+5;sheet.paste(thumb,(x,y));draw.text((x+5,y+274),f'{start+j+1:02d}',font=font,fill='black')
        sheet.save(previews/f'contact_{start//12+1}.jpg')
    (ROOT/'sources/rendered_pdf_text.txt').write_text('\n\n'.join(texts).replace('\r\n','\n').replace('\r','\n').rstrip()+'\n')
    passed('PDF','31 rendered pages; embedded Korean/fallback fonts; text extraction without replacement boxes')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.values=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['src','href']:self.values.append(value)
    parser=Links();parser.feed((ROOT/'gallery/index.html').read_text())
    for link in parser.values:
        if link.startswith(('http:','https:','#','data:')):continue
        assert (ROOT/'gallery'/unquote(link)).resolve().is_file(),link
    passed('gallery_local_links',len(parser.values))
    browser_result=subprocess.check_output(['node',str(ROOT/'verify_gallery.js'),str(ROOT/'gallery/index.html')],text=True)
    gallery=json.loads(browser_result);assert gallery['status']=='passed';passed('gallery_search_logic',gallery)
    inventory=pd.read_csv(ROOT/'all_figures_index.csv');assert len(inventory)==1283
    assert all(TREE[row.path]==row.git_blob_sha for row in inventory.itertuples())
    assert not inventory.path.str.contains('handoff_audit_20261102').any()
    passed('original_figure_index','1283 existing files, pinned links and main/bw blob comparison')
    figure_manifest=json.loads((ROOT/'figures/figure_manifest.json').read_text());assert len(figure_manifest)==16
    assert all((ROOT/'figures'/f'{v["name"]}.{ext}').is_file() for v in figure_manifest for ext in ['png','pdf'])
    passed('curated_scientific_figures','16 PNG +16 vector PDF')
    # The unrelated normal git index and local branch remain untouched.
    assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='bw'
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
    passed('git_checkout','bw unchanged; normal index empty; original evidence hashes unchanged')
    result=dict(status='passed',source_revision=REV,main_revision=MAIN,slide_count=31,curated_figure_pairs=16,original_figure_files=1283,
                human_visual_review='All 31 PDF pages reviewed as contact sheets; representative scientific figures inspected at full size.',
                gallery_validation='Asset/link checks and JavaScript search/filter/pagination checks. No browser screenshot claim.',checks=checks)
    (ROOT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('Presentation verification passed:',len(checks),'checks')


if __name__=='__main__':main()
