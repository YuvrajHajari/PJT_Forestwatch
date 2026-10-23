// Published CA / CA-ANN studies this project is validated against,
// reused from the literature review (ForestWatch_India_IEEE_Paper /
// Project_Report, Chapter 2 and its results-comparison chapter). Kappa
// figures are exactly as published by each study.
export const OUR_PIPELINE = {
  name: "This work",
  area: "Chennai, Bengaluru, Delhi, Kolkata",
  method: "FVI (custom spectral formula) + CA-ANN (Markov rate + trained MLP) + Clay v1.5 frozen embeddings",
  horizon: "1-year, walk-forward",
  kappa: 0.78,
  note: "4 cities validated, external Hansen GFC check, deployed as a live system. None of the comparison studies do all three.",
};

export const PAPERS = [
  {
    name: "Baig et al. (2022)",
    area: "Selangor, Malaysia",
    method: "SVM classification + CA-ANN",
    horizon: "Unspecified",
    kappa: 0.72,
  },
  {
    name: "Agustina et al. (2022)",
    area: "Cirebon, Indonesia",
    method: "Cellular automata, no neural component",
    horizon: "10-year, held-out",
    kappa: 0.786,
  },
  {
    name: "Salehi Asfichi et al. (2018), mixed",
    area: "Tehran, Iran",
    method: "CA-ANN, mixed-model variant",
    horizon: "10-year, held-out",
    kappa: 0.76,
  },
  {
    name: "Salehi Asfichi et al. (2018), raster",
    area: "Tehran, Iran",
    method: "CA-ANN, common-raster variant",
    horizon: "10-year, held-out",
    kappa: 0.705,
  },
  {
    name: "Lukas et al. (2023)",
    area: "Upper Omo-Gibe, Ethiopia",
    method: "CA-ANN (MOLUSCE), bidirectional forest class",
    horizon: "~31-year, held-out",
    kappa: 0.82,
  },
  {
    name: "Legdou et al. (2020)",
    area: "Middle Atlas, Morocco",
    method: "Random forest + CA, bidirectional forest strata",
    horizon: "~28-year, held-out",
    kappa: 0.721,
  },
];
