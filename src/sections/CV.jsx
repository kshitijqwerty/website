import CVPreview from "../components/CVPreview";
import cvData from "../data/cv.json";

export default function CV() {
  return (
    <section id="cv" className="py-24">
      <div className="flex flex-wrap items-baseline justify-between gap-4 mb-8">
        <h2 className="text-4xl font-bold font-heading">CV</h2>

        <div className="flex flex-wrap gap-3">
          <a
            href={`/kshitij_cv.pdf?v=${cvData._version}`}
            download
            className="px-5 py-2.5 rounded-2xl border border-neutral-700 hover:border-neutral-500 transition-colors text-sm"
          >
            Download PDF
          </a>
        </div>
      </div>

      <CVPreview data={cvData} />
    </section>
  );
}
