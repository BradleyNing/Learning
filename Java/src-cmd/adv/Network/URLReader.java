import java.net.*;
import java.io.*;
public class URLReader {
	try {
		public static void main(String[] args) throws Exception {
			URL cs = new URL("http://www.surrey.ac.uk");
			BufferedReader in  = new BufferedReader(new InputStreamReader(cs.openStream()));
			String inputLine;
			System.out.println(in);
			while((inputLine = in.readLine()) != null) {
				System.out.println(inputLine);
			}

			in.close();
		} catch(Exception e) {System.out.println(e);}
	}
}
