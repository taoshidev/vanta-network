import { useState, useEffect } from "react";
import { Center, Loader } from "@mantine/core";

import { MinerData } from "../../types";
import { getMinerData } from "../../lib";

import { ErrorBoundary } from "../ErrorBoundary";
import { ErrorFallback } from "../ErrorFallback";
import { Main } from "../Main";

import "./App.css";
import { isEmpty } from "lodash";

export const App = () => {
  const [data, setData] = useState<MinerData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);

      try {
        const minerData = await getMinerData();

        setData(minerData);
      } catch (error: unknown) {
        if (error instanceof Error) {
          setError(error.message);
          throw new Error(error.message);
        } else {
          setError("An unknown error occurred"); // Optional: Handle non-Error objects
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Center>
        <Loader color="orange" type="dots" />
      </Center>
    );
  }

  if (isEmpty(data?.positions) || isEmpty(data?.statistics.data)) {
    return <Center>No data available</Center>;
  }

  return (
    <ErrorBoundary
      fallback={<ErrorFallback error={new Error(error as string)} />}
    >
      <Main data={data as MinerData} />
    </ErrorBoundary>
  );
};
