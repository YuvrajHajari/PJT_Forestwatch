import { useEffect, useState } from "react";
import { api } from "./api";
import Landing from "./components/Landing";
import FetchImages from "./components/intro/FetchImages";
import MaskExplainer from "./components/intro/MaskExplainer";
import AppNav from "./components/AppNav";
import SingleYearSection from "./components/sections/SingleYearSection";
import CompareSection from "./components/sections/CompareSection";
import PredictSection from "./components/sections/PredictSection";
import MoreFeaturesSection from "./components/sections/MoreFeaturesSection";
import BigQuestionSection from "./components/sections/BigQuestionSection";
import Footer from "./components/Footer";

export default function App() {
  const [stage, setStage] = useState("landing"); // landing | fetch | mask | app
  const [cities, setCities] = useState([]);
  const [city, setCity] = useState(null);
  const [years, setYears] = useState([]);
  const [introYear, setIntroYear] = useState(null);
  const [section, setSection] = useState("single");
  const [compareCityB, setCompareCityB] = useState("");

  useEffect(() => {
    api.cities().then((d) => setCities(d.cities));
  }, []);

  const handleSelectCity = (key) => {
    setCity(key);
    setStage("fetch");
    api.years(key).then((d) => {
      setYears(d.years);
      const other = cities.find((c) => c.key !== key);
      setCompareCityB(other ? other.key : "");
    });
  };

  const handleChangeCity = () => {
    setStage("landing");
    setCity(null);
    setSection("single");
  };

  const cityLabel = cities.find((c) => c.key === city)?.label || city || "";

  if (stage === "landing" || !city) {
    return <Landing onSelectCity={handleSelectCity} />;
  }

  if (stage === "fetch") {
    if (years.length === 0) {
      return (
        <div className="intro-step intro-loading">
          <p>Loading {cityLabel}&rsquo;s fetched years&hellip;</p>
        </div>
      );
    }
    return (
      <div className="intro-shell">
        <FetchImages
          city={city}
          cityLabel={cityLabel}
          years={years}
          onContinue={() => {
            setIntroYear(years[years.length - 1]);
            setStage("mask");
          }}
        />
      </div>
    );
  }

  if (stage === "mask") {
    return (
      <div className="intro-shell">
        <MaskExplainer
          city={city}
          cityLabel={cityLabel}
          year={introYear}
          onContinue={() => setStage("app")}
        />
      </div>
    );
  }

  return (
    <>
      <AppNav
        cityLabel={cityLabel}
        section={section}
        onNavigate={setSection}
        onChangeCity={handleChangeCity}
      />
      <main className="app-main">
        {section === "single" && <SingleYearSection city={city} cityLabel={cityLabel} years={years} />}
        {section === "compare" && <CompareSection city={city} cityLabel={cityLabel} years={years} />}
        {section === "predict" && <PredictSection city={city} cityLabel={cityLabel} />}
        {["risk", "conservation", "citycompare"].includes(section) && (
          <MoreFeaturesSection
            sub={section}
            city={city}
            cityLabel={cityLabel}
            years={years}
            cities={cities}
            compareCityB={compareCityB}
            setCompareCityB={setCompareCityB}
          />
        )}
        {section === "bigquestion" && <BigQuestionSection />}
      </main>
      <Footer />
    </>
  );
}
