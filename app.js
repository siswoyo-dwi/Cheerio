import 'dotenv/config';
import { JsonOutputFunctionsParser } from "langchain/output_parsers";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI,OpenAIEmbeddings } from "@langchain/openai";
import {
    RunnableSequence,
    RunnablePassthrough,
  } from "@langchain/core/runnables";
const loader = new CheerioWebBaseLoader(
  "https://www.olx.co.id/jakarta-dki_g2000007/q-avanza-2020?sorting=asc-price"
);

const docs = await loader.load();
// console.log(docs);


docs[0].pageContent = docs[0].pageContent.substring(0, 8000);

const vectorStore = await HNSWLib.fromDocuments(
    docs,
    new OpenAIEmbeddings()
    );

// await vectorStore.save(directory);
// const vectorStore = await HNSWLib.load(directory, embeddings);

// vectorStore.similaritySearch()
const retriever = vectorStore.asRetriever({k:1});
// console.log(retriever);
const prompt =
PromptTemplate.fromTemplate(`please analyze this cars catalog from this context:
{konteks}

if you found something like Rp 115.000.0002015 it means the car prize is Rp 115.000.000 and car year is 2015
question: {pertanyaan}


`);



const parser = new JsonOutputFunctionsParser();

// Define the function schema
const extractionFunctionSchema = {
  name: "extractor",
  description: "Extracts fields from the context.",
  parameters: {
    type: "object",
    properties: {
    //   jenis: {
    //     type: "string",
    //     description: "jenis material",
    //   },
      harga_terendah: {
        type: "number",
        description: "car lowest price",
      },
      car_type: {
        type: "string",
        description: "car type",
      },
      car_year: {
        type: "string",
        description: "car year",
      },
    //   jawaban: {
    //     type: "string",
    //     description: "respon natural language",
    //   },
    },
    required: ["harga_terendah","car_type", "car_year"],
  },
};

const model = new ChatOpenAI({model: "gpt-4o-mini" });
const runnable = model
  .bind({
    functions: [extractionFunctionSchema],
    function_call: { name: "extractor" },
  })
  .pipe(parser);
const chain = RunnableSequence.from([
{
  konteks: retriever.pipe(formatDocumentsAsString),
  pertanyaan: new RunnablePassthrough(),
},
prompt,
runnable
])
;

const result = await chain.invoke("berapa harga terendah dan tipe mobil dari list mobil tersebut?");

console.log(result);