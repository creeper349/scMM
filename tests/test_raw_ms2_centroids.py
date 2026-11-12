from types import SimpleNamespace

import numpy as np
import pyopenms as oms

from scMM.file.thermo_raw import load_thermo_raw


def test_ms2_uses_official_centroids_and_ms1_keeps_profile(monkeypatch, tmp_path):
    calls = []
    def centroid(n, flag):
        calls.append(n)
        return SimpleNamespace(Masses=[60.0815,86.0969,184.0731],Intensities=[10.,50.,100.])
    raw = SimpleNamespace(
        IsOpen=True,IsError=False,InAcquisition=False,RunHeaderEx=SimpleNamespace(FirstSpectrum=1,LastSpectrum=2),
        FileHeader=SimpleNamespace(CreationDate=None),SelectInstrument=lambda *_:None,
        GetInstrumentData=lambda:SimpleNamespace(Model='Test',Name='Test'),
        GetFilterForScanNumber=lambda n:SimpleNamespace(MSOrder=n),
        GetScanStatsForScanNumber=lambda _:SimpleNamespace(IsCentroidScan=False,SpectrumPacketType='FtProfile'),
        GetCentroidStream=centroid,
        GetSegmentedScanFromScanNumber=lambda *_:SimpleNamespace(Positions=[100.,100.001,100.002],Intensities=[0.,100.,0.]),
        GetScanEventForScanNumber=lambda _:SimpleNamespace(GetReaction=lambda _:SimpleNamespace(PrecursorMass=826.5381,CollisionEnergy=30.)),
        RetentionTimeFromScanNumber=lambda _:1.,Dispose=lambda:None,
    )
    monkeypatch.setattr('scMM.file.thermo_raw._load_api',lambda:(SimpleNamespace(FileFactory=lambda _:raw),SimpleNamespace(MS=1)))
    exp,_=load_thermo_raw(str(tmp_path/'test.raw'),centroid_ms2=True)
    assert calls == [2]
    assert exp[0].getType() == oms.SpectrumSettings.SpectrumType.PROFILE
    assert exp[1].getType() == oms.SpectrumSettings.SpectrumType.CENTROID
    np.testing.assert_allclose(exp[1].get_peaks()[0],[60.0815,86.0969,184.0731])
