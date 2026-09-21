import { useState, useCallback, useEffect } from "react";
import { api } from "./api";
import Masthead from "./components/Masthead";
import Controls from "./components/Controls";
import Stage from "./components/Stage";
import SidePanel from "./components/SidePanel";
import Footer from "./components/Footer";

const MODES = [
  { key: "single", label: "Single year" },
  { key: "compare", label: "Compare years" },
  { key: "predict", label: "Project forward" },
  { key: "risk", label: "Risk heatmap" },
  { key: "conservation", label: "Conservation priority" },
  { key: "hansen", label: "Hansen validation" },
  { key: "timelapse", label: "Time-lapse" },
  { key: "citycompare", label: "City vs city" },
];

// Fixed prediction horizons matching the backend's own multi-year
// validation test points (train_ca_ann.py's HORIZONS_TO_TEST) -- these
// are deliberately not city-dependent, unlike the year lists below.
const TARGET_YEARS = ["2028", "2031", "2034"];

export default function App() {
  // Cities and years are fetched from the backend rather than
  // hardcoded, so adding a new city (edit cities_config.py, fetch its
  // imagery) never requires touching this frontend at all.
  const [cities, setCities] = useState([]);
  const [city, setCity] = useState("");
  const [years, setYears] = useState([]);

  const [mode, setMode] = useState("single");
  const [year, setYear] = useState("");
  const [yearA, setYearA] = useState("");
  const [yearB, setYearB] = useState("");
  const [targetYear, setTargetYear] = useState(TARGET_YEARS[0]);
  const [hansenYear, setHansenYear] = useState("");
  const [compareCityB, setCompareCityB] = useState("");

  const [satView, setSatView] = useState("satellite");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trendData, setTrendData] = useState({}); // { [city]: { [year]: ha } }
  const [priorityBackdrop, setPriorityBackdrop] = useState(null); // satellite img for conservation mode markers

  // Load the city list once on mount, and default compareCityB to the
  // second available city right away (not via a separate reactive
  // effect chasing state changes).
  useEffect(() => {
    api
      .cities()
      .then((d) => {
        setCities(d.cities);
        if (d.cities.length > 0) setCity(d.cities[0].key);
        if (d.cities.length > 1) setCompareCityB(d.cities[1].key);
      })
      .catch((e) => setError(e.message));
  }, []);

  // Reload the available years whenever the selected city changes, and
  // reset year selections to sensible defaults for that city's actual
  // fetched range (a newly-added city may have fewer years available
  // than an established one).
  useEffect(() => {
    if (!city) return;
    api
      .years(city)
      .then((d) => {
        setYears(d.years);
        if (d.years.length > 0) {
          const last = d.years[d.years.length - 1];
          const first = d.years[0];
          setYear(last);
          setYearA(first);
          setYearB(last);
          setHansenYear(last);
        } else {
          setYear("");
          setYearA("");
          setYearB("");
          setHansenYear("");
        }
      })
      .catch((e) => setError(e.message));
  }, [city]);

  const resetForNewRun = () => {
    setError(null);
    setLoading(true);
  };

  const runSingle = useCallback(async () => {
    resetForNewRun();
    try {
      const d = await api.analyze(city, year);
      setData(d);
      setTrendData((prev) => ({
        ...prev,
        [city]: { ...(prev[city] || {}), [year]: d.forest_area_ha },
      }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city, year]);

  const runCompare = useCallback(async () => {
    if (yearA === yearB) {
      setError("Choose two different years");
      return;
    }
    resetForNewRun();
    try {
      const [cmp, a, b] = await Promise.all([
        api.compare(city, yearA, yearB),
        api.analyze(city, yearA),
        api.analyze(city, yearB),
      ]);
      setData({ ...cmp, yearAOverlay: a.overlay_url, yearBOverlay: b.overlay_url });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city, yearA, yearB]);

  const runPredict = useCallback(async () => {
    resetForNewRun();
    try {
      const d = await api.predict(city, targetYear);
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city, targetYear]);

  const runRisk = useCallback(async () => {
    resetForNewRun();
    try {
      const d = await api.risk(city);
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city]);

  const runConservation = useCallback(async () => {
    resetForNewRun();
    try {
      const [priority, latest] = await Promise.all([
        api.conservationPriority(city),
        api.years(city),
      ]);
      const latestYear = latest.years[latest.years.length - 1];
      const backdrop = await api.analyze(city, latestYear);
      setData(priority);
      setPriorityBackdrop(backdrop.overlay_url);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city]);

  const runHansen = useCallback(async () => {
    resetForNewRun();
    try {
      const d = await api.validateHansen(city, hansenYear);
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city, hansenYear]);

  // No backend call needed -- every year's raw satellite image already
  // exists on disk once satellite_fetch.py has run. "Running" this mode
  // just packages the already-known years list so Stage/SidePanel follow
  // the same data-driven show/hide pattern as every other mode.
  const runTimelapse = useCallback(async () => {
    resetForNewRun();
    try {
      if (years.length === 0) {
        throw new Error("No years of imagery fetched for this city yet.");
      }
      setData({ city, years });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city, years]);

  const runCityCompare = useCallback(async () => {
    if (!compareCityB || compareCityB === city) {
      setError("Choose a different second city");
      return;
    }
    resetForNewRun();
    try {
      const [a, b] = await Promise.all([
        api.analyze(city, year),
        api.analyze(compareCityB, year),
      ]);
      setData({ cityA: { key: city, ...a }, cityB: { key: compareCityB, ...b } });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [city, compareCityB, year]);

  const runners = {
    single: runSingle,
    compare: runCompare,
    predict: runPredict,
    risk: runRisk,
    conservation: runConservation,
    hansen: runHansen,
    timelapse: runTimelapse,
    citycompare: runCityCompare,
  };

  const handleRun = () => {
    setData(null);
    runners[mode]();
  };

  const handleSetMode = (m) => {
    setMode(m);
    setData(null);
    setError(null);
  };

  const handleSetCity = (c) => {
    setCity(c);
    setData(null);
    setError(null);
    // If the newly-selected primary city collides with the current
    // "compare against" city, pick a different one right here -- this
    // is the actual event that can cause a collision, so fixing it here
    // avoids a separate effect that would just re-derive the same thing
    // a moment later.
    if (c === compareCityB) {
      const other = cities.find((ct) => ct.key !== c);
      if (other) setCompareCityB(other.key);
    }
  };

  return (
    <>
      <Masthead />
      <div className="masthead-rule" />
      <Controls
        modes={MODES}
        mode={mode}
        setMode={handleSetMode}
        cities={cities}
        city={city}
        setCity={handleSetCity}
        compareCityB={compareCityB}
        setCompareCityB={setCompareCityB}
        year={year}
        setYear={setYear}
        yearA={yearA}
        setYearA={setYearA}
        yearB={yearB}
        setYearB={setYearB}
        targetYear={targetYear}
        setTargetYear={setTargetYear}
        hansenYear={hansenYear}
        setHansenYear={setHansenYear}
        years={years}
        targetYears={TARGET_YEARS}
        onRun={handleRun}
        loading={loading}
        error={error}
      />
      <div className="page">
        <Stage
          mode={mode}
          city={city}
          data={data}
          loading={loading}
          satView={satView}
          setSatView={setSatView}
          priorityBackdrop={priorityBackdrop}
        />
        <SidePanel mode={mode} data={data} trendData={trendData[city] || {}} />
      </div>
      <Footer />
    </>
  );
}
