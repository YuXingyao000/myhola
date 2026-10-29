"""完整W&B历史：比较同一DDPM目标，并单列DPO偏好目标。"""

from pathlib import Path
import json
import numpy as np
from src.brepnet.eval.photo_report import setup_plot

ROOT = Path(__file__).resolve().parent


def pairs(rows, key):
    return np.array([(row['trainer/global_step'] + 1, row[key]) for row in rows
                     if row.get(key) is not None and np.isfinite(row[key])], dtype=float)


def main():
    plt = setup_plot()
    out = ROOT / 'visualizations'
    spatial = json.loads((out / 'spatial_history.json').read_text())
    dpo = json.loads((out / 'photo_dpo_history.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.subplots_adjust(left=.075, right=.96, top=.86, bottom=.12, hspace=.42, wspace=.25)
    fig.suptitle('停止前检查：空间条件模型与照片 DPO 的训练曲线', fontsize=20, weight='bold', y=.97)
    fig.text(.075, .91, '完整 W&B 历史；训练使用 epoch DDPM MSE。DPO 在5000步的最后一点仅含8步，不能视作一个完整epoch。', fontsize=11)
    summary = {}
    for index, (name, rows, title) in enumerate((('spatial', spatial, '空间条件：从头训练'), ('photo_dpo', dpo, '照片310k → DPO5000'))):
        train, val = pairs(rows, 'train/ddpm_loss'), pairs(rows, 'val/loss')
        ax = axes[0, index]
        ax.plot(train[:, 0], train[:, 1], label='训练 DDPM MSE（epoch）', color='#2469b0', lw=1.8)
        ax.plot(val[:, 0], val[:, 1], label='完整 validation MSE', color='#d37822', lw=1.8, marker='o', ms=3)
        ax.set(title=title, xlabel='optimizer step', ylabel='epsilon MSE', yscale='log')
        ax.legend(fontsize=10)
        best = val[np.argmin(val[:, 1])]
        ax.axvline(best[0], ls='--', color='#687786', lw=1)
        summary[name] = {'validation_points':len(val), 'best_step':int(best[0]), 'best_val_mse':best[1],
                         'last_monitored_step':int(val[-1,0]), 'last_val_mse':val[-1,1],
                         'rise_from_best_percent':(val[-1,1]/best[1]-1)*100,
                         'last_train_ddpm_mse':train[-1,1],
                         'last_train_point_partial_epoch':name == 'photo_dpo',
                         'last_complete_epoch_train_ddpm_mse':train[-2,1] if name == 'photo_dpo' else train[-1,1]}
    train, val = pairs(spatial,'train/ddpm_loss'), pairs(spatial,'val/loss')
    ax=axes[1,0]
    for values,label,color in ((train,'训练 DDPM MSE','#2469b0'),(val,'validation MSE','#d37822')):
        values=values[values[:,0]>=15000]
        ax.plot(values[:,0],values[:,1],label=label,color=color,lw=1.8)
    best=summary['spatial'];ax.axvline(best['best_step'],ls='--',color='#687786',lw=1)
    ax.scatter([best['best_step']],[best['best_val_mse']],color='#d37822',zorder=4)
    ax.set(title=f"空间模型后期：验证比最低点回升 {best['rise_from_best_percent']:.1f}%",xlabel='optimizer step',ylabel='epsilon MSE',ylim=(.025,.075))
    ax.annotate(f"best {best['best_step']:,}\n{best['best_val_mse']:.5f}",
                (best['best_step'],best['best_val_mse']),xytext=(27000,.066),arrowprops={'arrowstyle':'->','color':'#687786'},fontsize=10)
    ax.legend(fontsize=10)
    ax=axes[1,1]
    pref=pairs(dpo,'train/preference_loss');acc=pairs(dpo,'train/preference_accuracy')
    ax.plot(pref[:,0],pref[:,1],label='训练 preference loss',color='#7457a5',marker='o',ms=3)
    ax.set(title='DPO偏好目标（与DDPM MSE分开）',xlabel='DPO optimizer step',ylabel='preference loss')
    twin=ax.twinx();twin.plot(acc[:,0],acc[:,1],label='训练 preference accuracy',color='#16877b',marker='o',ms=3)
    twin.set_ylabel('preference accuracy');twin.set_ylim(0,1)
    ax.legend(loc='upper left',fontsize=9);twin.legend(loc='lower right',fontsize=9)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.text(.075,.045,'空间模型出现泛化差距扩大；仍需比较 best / last 的实际 B-Rep。DPO验证后期回升，也不直接等同于合法率下降。',fontsize=11)
    fig.savefig(out/'training_curves.png',dpi=170)
    fig.savefig(out/'training_curves.svg')
    plt.close(fig)
    (ROOT/'CURVE_REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
